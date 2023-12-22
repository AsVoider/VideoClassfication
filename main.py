import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lab3_data_scratch
import my_model, pretrained_model
import torch.nn as nn
import train_eva
import visual_history
import argparse
from torch.utils.data import DataLoader as dtl

num_classes = 10
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_frames = 40  # You can adjust this to balance speed and accuracy
img_size = (128, 128)  # You can adjust this to balance speed and accuracy
num_workers = 4
last_weights = 'last_weights.pt'
best_weights = 'best_weights.pt'
transform = A.Compose(
    [
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(),
        ToTensorV2()
    ]
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="select model")
    parser.add_argument('--model_select', type=str, default="my_model",
                        help='model select from my_model, pre_trained')
    parser.add_argument('--lr_select', type=str, default="re",
                        help='lr method select from \'RE\' or \'COS\'')
    parser.add_argument('--pt_select', type=str, default="last",
                        help='pt last ? best ? acc')
    parser.add_argument('--train', type=str, default='yes',
                        help="yes or no / train or only test")
    args = parser.parse_args()
    train_, val_, test_, full_ = lab3_data_scratch.transform_data(num_classes, batch_size, num_frames, num_workers,
                                                                  './data', transform)
    model = my_model.MyModel(num_classes=num_classes, hidden_size=128, num_lstm=2) if args.model_select == "my_model" \
        else pretrained_model.ClassificationModel(num_classes=num_classes, hidden_size=128, num_lstm_layers=2)
    loss_fn = nn.CrossEntropyLoss()
    if args.train == 'yes':
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, mode='min', patience=3,
                                                               verbose=True) if args.lr_select == 'RE' \
            else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 100, 1, 1e-6, verbose=True)
        model, his = train_eva.train(model, train_, loss_fn, opt, weights=None, epochs=50, validation_data=val_,
                                     save_best_weights_path=best_weights, save_last_weights_path=last_weights,
                                     device=device, validation_split=None, steps_per_epoch=100, scheduler=scheduler)
        visual_history.visualize_history(his)
    else:
        test_ = dtl(full_, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loss, test_acc = train_eva.evaluate(model, weights=last_weights, val_data=test_, loss_fn=loss_fn,
                                             device='cuda', verbose=1)
    print(f'Loss: {test_loss : .3f}, Acc: {test_acc: .3f}')
