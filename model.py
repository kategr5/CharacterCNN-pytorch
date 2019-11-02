import torch.nn as nn

class Model(nn.Module):
    def __init__(self,dataset='<dataset>'):
        super(Model, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(70, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 34, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
         )
        if dataset == 'AGNews':
            self.fc3 = nn.Linear(1024, 4)  # number of classes
        elif dataset == '20Newsground':
            self.fc3 = nn.Linear(1024, 20)
        self.soft = nn.Softmax(dim=1)

        #self.weight_init(mean=0.0,std=0.05)

    '''def weight_init(self, mean=0.0, std=0.05):
        for mod in self.modules():
            if isinstance(mod, nn.Conv1d) or isinstance(mod, nn.Linear):
                mod.weight.data.normal_(mean, std)'''


    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        out = self.layer5(out)
        #print(out.shape)
        out = self.layer6(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.soft(out)
        return out
