from utility import *
from prune import *
from linear_quantization import *

if __name__ == "__main__":
    model, dataloader = get_model_and_dataloader()

    # First Channel Prune
    channel_model = copy.deepcopy(model)
    channel_pruner = ChannelPruner()
    channel_pruner.prune(channel_model, 0.3)
    channel_pruner.finetune(channel_model, dataloader)

    # Then Fine Grained Prune
    fg_model = copy.deepcopy(channel_model)
    fg_pruner = FineGrainedPruner()
    fg_pruner.prune(fg_model, dataloader)
    fg_pruner.finetune(fg_model, dataloader)
    evaluate_and_print_metrics(fg_model, dataloader, "Mixed Prune model", count_nonzero_only=True)

    # Finally 8 bit linear quantiziation
    bitwidth = 8
    model_cp = copy.deepcopy(fg_model)

    linear_qnt = LinearQuantizer(model_cp, dataloader, bitwidth=bitwidth)
    quantized_model = linear_qnt.quantize()

    evaluate_and_print_metrics(quantized_model, dataloader, "Mixed Prune & Qnt model", bitwidth=8, extra_preprocess=[extra_preprocess], count_nonzero_only=True)
