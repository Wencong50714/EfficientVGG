import torch
from vgg import *
from matplotlib import pyplot as plt
from data import *

def visualize_output(dataset):
    plt.figure(figsize=(20, 10))
    for index in range(40):
        image, label = dataset["test"][index]

        # Model inference
        model.eval()
        with torch.inference_mode():
            pred = model(image.unsqueeze(dim=0).cuda())
            pred = pred.argmax(dim=1)

        # Convert from CHW to HWC for visualization
        image = image.permute(1, 2, 0)

        # Convert from class indices to class names
        pred = dataset["test"].classes[pred]
        label = dataset["test"].classes[label]

        # Visualize the image
        plt.subplot(4, 10, index + 1)
        plt.imshow(image)
        plt.title(f"pred: {pred}" + "\n" + f"label: {label}")
        plt.axis("off")
        plt.show()

if __name__ == '__main__':
    
    model_path = "models/model.pth"
    if not os.path.exists(model_path):
        print("Warning: Model file not found at", model_path)
        print("Please train the model first!")
        exit(1)
    model = VGG.load(path=model_path)

    dataset, dataflow = prepare_data()

    # Calculate and output model accuracy on test set
    model.eval()
    correct = 0
    total = 0
    
    with torch.inference_mode():
        for images, labels in dataflow["test"]:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Option: Visualize some predictions
    # visualize_output(dataset)
