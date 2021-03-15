import timm
import torch
from torchsummary import summary

class Pretrained_Model:
    def __init__(self, shape, num_outputs):
        self.shape = shape
        self.num_outputs = num_outputs

    def build(self):
        # instantiate pre-trained model
        #self.m = timm.create_model('efficientnet_b0', pretrained=True)
        #self.m = timm.create_model('gernet_s', pretrained=True)
        #self.m = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
        # self.m = timm.create_model('tf_efficientnet_lite0', pretrained=True)
        #self.m = timm.create_model('semnasnet_075', pretrained=False)
        self.m = timm.create_model('efficientnet_lite0', pretrained=True)

        # remove the last layer of the pretrained model
        # 'classifier' is the name of the final layer of the model
        num_final_inputs = self.m.classifier.in_features
        self.m.classifier = torch.nn.Linear(num_final_inputs, self.num_outputs)
        '''
        self.m.classifier = torch.nn.Sequential(
            #torch.nn.Dropout(0.1),
            torch.nn.Linear(1280, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(),
            #torch.nn.Dropout(0.1),
            #torch.nn.Linear(num_final_inputs, self.num_outputs)
            torch.nn.Linear(500, self.num_outputs)
        )
        '''

        # GeNet - GPU efficient net (gernet_s)
        # num_final_inputs = self.m.head.fc.in_features
        # self.m.head.fc = torch.nn.Linear(num_final_inputs, self.num_outputs)

        print(self.m)
        summary(self.m.cuda(), (3, 224, 224))

        # return model
        return self.m

    def print_summary(self):
        if torch.cuda.is_available():
            summary(self.m.cuda(), (3, 224, 224))
        else:
            summary(self.m, (3, 224, 224))

if __name__ == '__main__':
    m = Pretrained_Model(shape=(360,640,3), num_outputs=128)
    model = m.build()
    print (model)

    m.print_summary()
