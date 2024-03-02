"""
Contains functions for training and testing a Pytorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
  """Trains a Pytorch model for a single epoch
  
  Turns a target Pytorch model to training mode and then
  runs throught all off the required training steps (forward pass,
  loss calculation, optimzer step)

  Args: 
    model: A Pytorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    optimizer: A Pytorch optimizer to help minimize the loss function.
    device: A target device to compute on ("cuda" or "cpu).

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). for example:

    (0.3412, 0.8232)
  """
  # put the model in training mode
  model.train()

  # setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop throught dataloader databatches
  for batch, (X, y) in enumerate(dataloader):
    # send data to target device
    X, y = X.to(device), y.to(device)

    # 1. perform forward pass
    y_pred = model(X)

    # 2. calculate and accumulate loss
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    # 3. optimizer zero grad
    optimizer.zero_grad()

    # 4. loss bacward
    loss.backward()

    # 5. optimzer step
    optimizer.step()

    # calculate accuracy across all batches
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class == y).sum().item()/len(y_pred)
  
  # adjust metrics to get average loss and acc
  train_loss /= train_loss/len(dataloader)
  train_acc /= train_acc/len(dataloader)

  return train_loss, train_acc


def test_step(model:torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  """ Tests Pytorch model for a single batch
  
  Turns a terget Pytorch model to "eval" model and then performs a
  forward pass on a testing dataset.

  Args:
    model: A Pytorch model to be tested.
    dataloader: A DataLoader intances for the model to be trained on.
    loss_fn: A Pytorch loss functon to calculate losss on the test data.
    device: A target device to compute on

  Returns: 
    A tuple of testing loss and tsting accuracy metrics.
    In tge form (test_loss, test_accuracy). For example:

    (0.0213, 0.8921)
  """
  # put model to eval mode
  model.eval()

  # setup loss and acc values
  test_loss, test_acc = 0, 0

  # turn on inference context manager
  with torch.inference_mode():
    # loop throught DataLoader batches
    for batch, (X, y) in enumerate(dataloader):
      # send data to device
      X, y = X.to(device), y.to(device)

      # 1. forward pass
      test_pred_logits = model(X)

      # 2. calculate and accumulate loss
      loss = loss_fn(test_pred_logits, y)
      test_loss += loss.item() 

      # calculate and accumulate accuracy
      test_pred_labels = test_pred_logits.argmax(dim=1)
      test_acc += ((test_pred_labels==y).sum().item() / len(test_pred_labels))

  # adjust metrrichs to get average loss and acc
  test_loss /= test_loss/len(dataloader)
  test_acc /= test_acc/len(dataloader)

  return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
  """Trains and tests a Pytorch model.

  Passes a target Pytorch model trhought train_step() and test_step()
  functions for a number of epochs, training and testing the model in
  the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args: 
    model: A Pytorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimzer: A Pytorch optimizer to minimize the loss function.
    loss_fn: A Pytorch loss function to calculate loss on both datasets.
    epochs: An inteher indeication how many epochs to train for.
    device: A terget device to compute on
  
  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metrics has a value in a list for 
    each epoch.
    In a form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]}
    For example if training for epochs=2:
    In a form: {train_loss: [2.3124, 1.2342],
                train_acc: [0.2312, 0.3245],
                test_loss: [1.3143, 1.2344],
                test_acc: [0.2452, 0.2131]}
  """
  # create empty result dict
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []}
  
  # loop throught training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                      dataloader=train_dataloader,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer,
                                      device=device)
    test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)
    # printout what happening
    print(
      f"Epoch: {epoch+1} | "
      f"train_loss : {train_loss:.4f} | "
      f"train_acc: {train_acc:.4f} | "
      f"test_loss: {test_loss:.4f} | "
      f"test_acc: {test_acc:.4f}"
    )

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
  
  return results