from utility import *
from prune import *
from quantization import *

if __name__ == "__main__":
    model, dataloader = get_model_and_dataloader()

    # Fine grained Prune
    fg_path = "models/fg_model.pth"
    if os.path.exists(fg_path):
        fg_model = VGG.load(path=fg_path).cuda()

    # Channel Prune
    fgc_model = copy.deepcopy(fg_model)
    channel_pruner = ChannelPruner()
    channel_pruner.prune(fgc_model, 0.3)
    channel_pruner.finetune(fgc_model, dataloader)
    modelname = "SA finegrained with channel prune"
    evaluate_and_print_metrics(fgc_model, dataloader, modelname)

    # 8 bit quantiziation
    bitwidth = 8
    model_cp = copy.deepcopy(fgc_model)

    quantizer = KMeansQuantizer(model_cp, bitwidth)
    quantizer.apply(model_cp, update_centroids=False)
    quantizer.finetune(model_cp, dataloader)
    modelname = f"{bitwidth}-bit k-means quantized model"
    evaluate_and_print_metrics(model_cp, dataloader, modelname, bitwidth)
