import torch
import tqdm
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import constrainScoreByWhole
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
__all__ = ["train", "validate", "modifier"]

def train(train_loader, model, criterion, optimizer, epoch, args, writer, weight_opt=None):
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    l = [losses, top1, top5]
    progress = ProgressMeter(
        len(train_loader),
        l,
        prefix=f"Epoch: [{epoch}]",
    )
    model.train()
    args.discrete = False
    for i, (images, target) in tqdm.tqdm(
            enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        target = target.squeeze()
        l, ol, gl, al, a1, a5, ll = 0, 0, 0, 0, 0, 0, 0
        if optimizer is not None:
            optimizer.zero_grad()
        if weight_opt is not None:
            weight_opt.zero_grad()
        for j in range(args.K):
            output = model(images)
            loss = criterion(output, target) / args.K
            loss.backward()
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            l = l + loss.item()
            a1 = a1 + acc1.item() / args.K
            a5 = a5 + acc5.item() / args.K
        losses.update(l, images.size(0))
        top1.update(a1, images.size(0))
        top5.update(a5, images.size(0))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        if optimizer is not None:
            optimizer.step()
        if weight_opt is not None:
            weight_opt.step()
        if args.conv_type == "ProbMaskConv":
            if not args.init_weights and not args.resume_train_weights and not args.finetuning:
                with torch.no_grad():
                    constrainScoreByWhole(model)
        if i % args.print_freq == 0:
            progress.display(i)
    progress.display(len(train_loader))
    progress.write_to_tensorboard(writer, prefix="train", global_step=epoch)
    return top1.avg, top5.avg



def validate(val_loader, model, criterion, args, writer, epoch):
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    losses_d = AverageMeter("Loss_d", ":.3f", write_val=False)
    top1_d = AverageMeter("Acc@1_d", ":6.2f", write_val=False)
    top5_d = AverageMeter("Acc@5_d", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [losses, top1, top5, losses_d, top1_d, top5_d], prefix="Test: "
    )
    if args.use_running_stats:
        model.eval()
    with torch.no_grad():
        for i, (images, target) in tqdm.tqdm(
                enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            target = target.squeeze()
            args.discrete = False
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            args.discrete = True
            output_d = model(images)
            loss_d = criterion(output_d, target)
            acc1_d, acc5_d = accuracy(output_d, target, topk=(1, 5))
            losses_d.update(loss_d.item(), images.size(0))
            top1_d.update(acc1_d.item(), images.size(0))
            top5_d.update(acc5_d.item(), images.size(0))
            if i % args.print_freq == 0:
                progress.display(i)
        progress.display(len(val_loader))
        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)
    return top1_d.avg, top5_d.avg, losses.avg


def modifier(args, epoch, model):
    return
