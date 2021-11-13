import math
import os

import torch
import torch.nn as nn
import torch.optim as optim

import give_valid_test
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8') #open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer
        word = ["<sos>"] + word
        word = word + ["<eos>"]

        if len(word) <= n_step:   #pad the sentence
            word = ["<pad>"]*(n_step+1-len(word)) + word

        for word_index in range(len(word)-n_step):
            input = [word2number_dict[n] for n in word[word_index:word_index+n_step]]  # create (1~n-1) as input
            target = word2number_dict[word[word_index+n_step]]  # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch # (batch num, batch size, n_step) (batch num, batch size)

def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  #open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))   #set to list

    word2number_dict = {w: i+2 for i, w in enumerate(word_list)}
    number2word_dict = {i+2: w for i, w in enumerate(word_list)}

    #add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"
    word2number_dict["<sos>"] = 2
    number2word_dict[2] = "<sos>"
    word2number_dict["<eos>"] = 3
    number2word_dict[3] = "<eos>"

    return word2number_dict, number2word_dict
class NewLSTM(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(NewLSTM, self).__init__()
        self.t = 0
        self.Wii = nn.Parameter(torch.randn(hidden_size, input_size, device=device))
        self.Whi = nn.Parameter(torch.randn(hidden_size, hidden_size, device=device))
        self.Wif = nn.Parameter(torch.randn(hidden_size, input_size, device=device))
        self.Whf = nn.Parameter(torch.randn(hidden_size, hidden_size, device=device))
        self.Wig = nn.Parameter(torch.randn(hidden_size, input_size, device=device))
        self.Whg = nn.Parameter(torch.randn(hidden_size, hidden_size, device=device))
        self.Wio = nn.Parameter(torch.randn(hidden_size, input_size, device=device))
        self.Who = nn.Parameter(torch.randn(hidden_size, hidden_size, device=device))
        self.bii = nn.Parameter(torch.ones(hidden_size, hidden_size,device=device))
        self.bhi = nn.Parameter(torch.ones(hidden_size, hidden_size,device=device))
        self.bif = nn.Parameter(torch.ones(hidden_size, hidden_size,device=device))
        self.bhf = nn.Parameter(torch.ones(hidden_size, hidden_size,device=device))
        self.big = nn.Parameter(torch.ones(hidden_size, hidden_size,device=device))
        self.bhg = nn.Parameter(torch.ones(hidden_size, hidden_size,device=device))
        self.bio = nn.Parameter(torch.ones(hidden_size, hidden_size,device=device))
        self.bho = nn.Parameter(torch.ones(hidden_size, hidden_size,device=device))
        self.f_list = []
        self.f_list.append(torch.zeros(hidden_size, hidden_size, device=device))
        self.i_list = []
        self.i_list.append(torch.zeros(hidden_size, hidden_size,device=device))
        self.g_list = []
        self.g_list.append(torch.zeros(hidden_size, hidden_size,device=device))
        self.o_list = []
        self.o_list.append(torch.zeros(hidden_size, hidden_size,device=device))
        self.c_list = []
        self.c_list.append(torch.zeros(hidden_size, hidden_size,device=device))
        self.h_list = []
        self.h_list.append(torch.zeros(hidden_size, hidden_size,device=device))

    def forward(self,X):
        for self.t in range(1,n_step):
            x = X[self.t-1,:,:]
            h = self.h_list[self.t - 1]
            i = torch.sigmoid(torch.matmul(self.Wii, x) + self.bii + torch.matmul(self.Whi, h) + self.bhi)
            self.i_list.append(i)
            f = torch.sigmoid(torch.matmul(self.Wif, x) + self.bif + torch.matmul(self.Whf, h) + self.bhf)
            self.f_list.append(f)
            g = torch.tanh(torch.matmul(self.Wig, x) + self.big + torch.matmul(self.Whg, h) + self.bhg)
            self.g_list.append(g)
            o = torch.sigmoid(torch.matmul(self.Wio, x) + self.bio + torch.matmul(self.Who, h) + self.bho)
            self.o_list.append(o)
            c = f * self.c_list[self.t - 1] + i * g
            self.c_list.append(c)
            h = o * torch.tanh(c)
            self.h_list.append(h)
        return self.o_list[-1],(self.h_list[-1],self.c_list[-1])


class operatelstm(nn.Module):
    def __init__(self):
        super(operatelstm, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.LSTM = NewLSTM(input_size=emb_size, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self,X):
        X = self.C(X)
        X = X.transpose(0, 1)
        X = X.transpose(1, 2)
        self.LSTM.i_list=[] #很重要！！！！！reset内部状态
        self.LSTM.i_list.append(torch.zeros(n_hidden,n_hidden,device=device))
        self.LSTM.f_list = []
        self.LSTM.f_list.append(torch.zeros(n_hidden, n_hidden, device=device))
        self.LSTM.g_list = []
        self.LSTM.g_list.append(torch.zeros(n_hidden, n_hidden, device=device))
        self.LSTM.h_list = []
        self.LSTM.h_list.append(torch.zeros(n_hidden, n_hidden, device=device))
        self.LSTM.c_list = []
        self.LSTM.c_list.append(torch.zeros(n_hidden, n_hidden, device=device))
        self.LSTM.o_list = []
        self.LSTM.o_list.append(torch.zeros(n_hidden, n_hidden, device=device))
        outputs, (_, _) = self.LSTM.forward(X)
        model = self.W(outputs) + self.b
        return model
def train_LSTMlm():
    model = operatelstm()
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    # Training
    batch_number = len(all_input_batch)
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            output = model.forward(input_batch)
            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))
            loss.backward()
            optimizer.step()

            count_batch += 1
        print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
              'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(data_root, word2number_dict, n_step)
        all_valid_batch = torch.LongTensor(all_valid_batch).to(device)  # list to tensor
        all_valid_target = torch.LongTensor(all_valid_target).to(device)

        total_valid = len(all_valid_target)*128  # valid and test batch size is 128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1

            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'loss =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch+1) % save_checkpoint_epoch == 0:
            torch.save(model, f'models/LSTMlm_model_epoch{epoch+1}.ckpt')

def LSTMlm_test(select_model_path):
    model = torch.load(select_model_path, map_location="cuda")  #load the selected model
    model.to(device)

    #load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(data_root, word2number_dict, n_step)
    all_test_batch = torch.LongTensor(all_test_batch).to(device)  # list to tensor
    all_test_target = torch.LongTensor(all_test_target).to(device)
    total_test = len(all_test_target)*128  # valid and test batch size is 128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('loss =','{:.6f}'.format(total_loss / count_loss),
          'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

if __name__ == '__main__':
    n_step = 5 # number of cells(= number of Step)
    n_hidden = 128 # number of hidden units in one cell
    batch_size = 128 # batch size
    learn_rate = 0.0005
    all_epoch = 5 #the all epoch for training
    emb_size = 256 #embeding size
    save_checkpoint_epoch = 5 # save a checkpoint per save_checkpoint_epoch epochs !!! Note the save path !!!
    data_root = 'penn_small'
    train_path = os.path.join(data_root, 'train.txt') # the path of train dataset

    print("print parameter ......")
    print("n_step:", n_step)
    print("n_hidden:", n_hidden)
    print("batch_size:", batch_size)
    print("learn_rate:", learn_rate)
    print("all_epoch:", all_epoch)
    print("emb_size:", emb_size)
    print("save_checkpoint_epoch:", save_checkpoint_epoch)
    print("train_data:", data_root)

    word2number_dict, number2word_dict = make_dict(train_path)
    #print(word2number_dict)

    print("The size of the dictionary is:", len(word2number_dict))
    n_class = len(word2number_dict)  #n_class (= dict size)

    print("generating train_batch ......")
    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    train_batch_list = [all_input_batch, all_target_batch]

    print("The number of the train batch is:", len(all_input_batch))
    all_input_batch = torch.LongTensor(all_input_batch).to(device)   #list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)
    # print(all_input_batch.shape)
    # print(all_target_batch.shape)
    all_input_batch = all_input_batch.reshape(-1, batch_size, n_step)
    all_target_batch = all_target_batch.reshape(-1, batch_size)

    print("\nTrain the LSTMLM……………………")
    train_LSTMlm()

    print("\nTest the LSTMLM……………………")
    select_model_path = "models/LSTMlm_model_epoch5.ckpt"
    LSTMlm_test(select_model_path)
