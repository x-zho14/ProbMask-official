import os
import pathlib
import random
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import copy
import numpy as np
from utils.conv_type import ProbMaskConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    freeze_model_subnet,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
    fix_model_subnet,
)
from utils.schedulers import get_policy, assign_learning_rate

from args import args
import importlib

import data
import models

def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    main_worker(args)


def main_worker(args):
    args.finetuning = False
    args.gpu = None
    train, validate, modifier = get_trainer(args)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model and optimizer
    model = get_model(args)
    model = set_gpu(args, model)
    torch.backends.cudnn.benchmark = True
    optimizer, weight_opt = get_optimizer(args, model)
    data = get_dataset(args)
    lr_policy = get_policy(args.lr_policy)(optimizer, args)
    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)
    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir
    writer = SummaryWriter(log_dir=log_base_dir)
    if args.evaluate:
        acc1, acc5, _ = validate(data.val_loader, model, criterion, args, None, epoch=args.start_epoch)
        return
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )
    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None
    # Save the initial state
    save_checkpoint(
        {
            "epoch": 0,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )
    pr_target = args.prune_rate
    ts = int(args.ts * args.epochs)
    te = int(args.te * args.epochs)
    pr_start = args.pr_start
    for epoch in range(args.start_epoch, args.epochs):
        lr_policy(epoch, iteration=None)
        if weight_opt is not None:
            assign_learning_rate(weight_opt, 0.5 * (1 + np.cos(np.pi * epoch / args.epochs)) * args.weight_opt_lr)
        if args.iterative:
            if epoch < ts:
                args.prune_rate = 1
            elif epoch < te:
                args.prune_rate = pr_target + (pr_start - pr_target)*(1-(epoch-ts)/(te-ts))**3
            else:
                args.prune_rate = pr_target
        if args.TA:
            args.T = 1 / ((1 - 0.03) * (1 - epoch / args.epochs) + 0.03)
        modifier(args, epoch, model)
        cur_lr = get_lr(optimizer)
        print("current lr: ", cur_lr)
        if weight_opt is not None:
            print("current weight lr: ", weight_opt.param_groups[0]["lr"])
        print("current temp: ", args.T)
        print("current prune rate: ", args.prune_rate)
        start_train = time.time()
        train_acc1, train_acc5 = train(data.train_loader, model, criterion, optimizer, epoch, args, writer=writer,
                                       weight_opt=weight_opt)
        train_time.update((time.time() - start_train) / 60)
        start_validation = time.time()
        acc1, acc5, losses = validate(data.val_loader, model, criterion, args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )
        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )
        if args.conv_type == "ProbMaskConv":
            for n, m in model.named_modules():
                if isinstance(m, ProbMaskConv) :
                    writer.add_scalar("pr/{}".format(n), m.clamped_scores.mean(), epoch)
                    print("average prune rate of this layer:", n, " ", m.clamped_scores.mean())
        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()
        print("best acc:%.2f, location:%d", best_acc1, log_base_dir)
    write_result_to_csv(
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        prune_rate=args.prune_rate,
        curr_acc1=acc1,
        curr_acc5=acc5,
        base_config=args.config,
        name=args.name + args.rep_count,
        epoch=str(args.epochs - 1),
        setting=str(args),
    )
    print("best_acc1: ", best_acc1)
    print("rep_count: ", args.rep_count)

    if args.finetune:
        best_acc1 = 0
        args.finetuning = True
        args.K = 1
        freeze_model_subnet(model)
        fix_model_subnet(model)
        args.batch_size = 128
        data = get_dataset(args)
        if args.sample_from_training_set:
            args.use_running_stats = False
            i = 0
            BESTACC1, BESTIDX = 0, 0
            BESTMODEL = None
            while i < 10:
                i += 1
                acc1, acc5, _ = validate(data.train_loader, model, criterion, args, None, epoch=args.start_epoch)
                if acc1 > BESTACC1:
                    BESTACC1 = acc1
                    BESTIDX = i
                    BESTMODEL = copy.deepcopy(model)
                print(BESTACC1, BESTIDX)
                for n, m in model.named_modules():
                    if hasattr(m, "scores"):
                        m.subnet = (torch.rand_like(m.scores) < m.clamped_scores).float()
            model = copy.deepcopy(BESTMODEL)
        args.use_running_stats = True

        args.lr = 0.001
        parameters = list(model.named_parameters())
        weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            weight_params,
            0.001,
            momentum=0.9,
            weight_decay=5e-4,
        )
        for epoch in range(0, 20):
            cur_lr = get_lr(optimizer)
            print("current lr: ", cur_lr)
            start_train = time.time()
            train_acc1, train_acc5 = train(
                data.train_loader, model, criterion, optimizer, epoch, args, writer=writer, weight_opt=None
            )
            train_time.update((time.time() - start_train) / 60)

            # evaluate on validation set
            start_validation = time.time()
            acc1, acc5, losses = validate(data.val_loader, model, criterion, args, writer, epoch)
            validation_time.update((time.time() - start_validation) / 60)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            best_acc5 = max(acc5, best_acc5)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            best_train_acc5 = max(train_acc5, best_train_acc5)

            save = ((epoch % args.save_every) == 0) and args.save_every > 0
            if is_best or save or epoch == args.epochs - 1:
                if is_best:
                    print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "best_acc1": best_acc1,
                        "best_acc5": best_acc5,
                        "best_train_acc1": best_train_acc1,
                        "best_train_acc5": best_train_acc5,
                        "optimizer": optimizer.state_dict(),
                        "curr_acc1": acc1,
                        "curr_acc5": acc5,
                    },
                    is_best,
                    filename=ckpt_base_dir / f"epoch_{epoch}.state",
                    save=save,
                )
            epoch_time.update((time.time() - end_epoch) / 60)
            progress_overall.display(epoch)
            progress_overall.write_to_tensorboard(
                writer, prefix="diagnostics", global_step=epoch
            )
            writer.add_scalar("test/lr", cur_lr, epoch)
            end_epoch = time.time()
            print("best acc:%.2f, location:%d", best_acc1, log_base_dir)


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"
    print(f"=> Parallelizing on {args.multigpu} gpus")
    torch.cuda.set_device(args.multigpu[0])
    args.gpu = args.multigpu[0]
    model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
        args.multigpu[0]
        )
    cudnn.benchmark = True
    return model

def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)
    return dataset


def get_model(args):
    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    print(model)
    return model


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    elif args.optimizer == "adam":
        if not args.train_weights_at_the_same_time:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            parameters = list(model.named_parameters())
            weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
            score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
            optimizer1 = torch.optim.Adam(
                score_params, lr=args.lr, weight_decay=args.weight_decay
            )
            optimizer2 = torch.optim.SGD(
                weight_params,
                args.weight_opt_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
            return optimizer1, optimizer2
    return optimizer, None


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    args.rep_count = "/"
    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1
        args.rep_count = "/" + str(rep_count)
        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Epoch, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5, "
            "Setting\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{epoch}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}, "
                "{setting}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()
