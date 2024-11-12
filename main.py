from model import SimpleCNN
from data_loader import get_data_loaders
from train import train_model, save_model
from utils import evaluate

def main():
    train_loader, test_loader = get_data_loaders(batch_size=64)

    model = SimpleCNN()

    trained_model = train_model(model, train_loader, num_epochs=5, lr=0.001)

    save_model(trained_model, path='model.pth')

    evaluate(trained_model, test_loader)

if __name__ == "__main__":
    main()
