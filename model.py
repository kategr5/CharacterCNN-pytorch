import torch.nn as nn

class Model(nn.Module):
    def __init__(self,dataset='<dataset>',size='<size>'):
        super(Model, self).__init__()
        if size=='big':
            self.feat_cnn=1024
            self.feat_fc=2048
        else: #default will be small
            self.feat_cnn=256
            self.feat_fc=1024
        self.layer1 = nn.Sequential(
            nn.Conv1d(70, self.feat_cnn, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(self.feat_cnn, self.feat_cnn, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(self.feat_cnn, self.feat_cnn, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(self.feat_cnn, self.feat_cnn, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(self.feat_cnn, self.feat_cnn, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv1d(self.feat_cnn, self.feat_cnn, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.feat_cnn * 34, self.feat_fc),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.feat_fc, self.feat_fc),
            nn.ReLU(),
            nn.Dropout(0.5)
         )

        if dataset == '20Newsground':
            self.fc3 = nn.Linear(self.feat_fc, 20)
        else: #if dataset == 'AGNews': AGNews is the default
            self.fc3 = nn.Linear(self.feat_fc, 4)  # number of classes
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
