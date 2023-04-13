import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import pickle


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)


class ProxyClient(fl.client.NumPyClient):
    def __init__(self, proxy_to):
        self.proxy_to = proxy_to

    def _proxy(self, fn):
        def proxied(*args, **kwargs):
            args_pickled = pickle.dumps(args)
            args_unpickled = pickle.loads(args_pickled)
            kwargs_pickled = pickle.dumps(kwargs)
            kwargs_unpickled = pickle.loads(kwargs_pickled)
            return_value = fn(*args_unpickled, **kwargs_unpickled)
            return_value_pickled = pickle.dumps(return_value)
            return_value_unpickled = pickle.loads(return_value_pickled)
            print("fn", fn,
                  "size of args", len(args_pickled),
                  "size of kwargs", len(kwargs_pickled),
                  "size of return_value", len(return_value_pickled))
            return return_value_unpickled
        return proxied

    def get_parameters(self, config):
        # Instantiate a new proxy_to class to test statelessness.
        return self._proxy(self.proxy_to().get_parameters)(config)

    def set_parameters(self, parameters):
        return self._proxy(self.proxy_to().set_parameters)(parameters)

    def fit(self, parameters, config):
        return self._proxy(self.proxy_to().fit)(parameters, config)

    def evaluate(self, parameters, config):
        return self._proxy(self.proxy_to().evaluate)(parameters, config)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        # Instantiate once per FlowerClient
        self.net = Net().to(DEVICE)
        self.trainloader, self.testloader = load_data()

    def get_parameters(self, config):
        print("==== GET_PARAMETERS")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        # uh-oh, this looks darn stateful to me. But, the pattern is normally
        # set_parameters immediately followed by a fit or evaluate, right? so we
        # could just batch them up? store the state on the client and only pass
        # it through when we call something that actually needs to access the
        # data, right?
        print("==== SET_PARAMETERS")
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("==== FIT")
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("==== EVALUATE")
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=ProxyClient(FlowerClient),
)
