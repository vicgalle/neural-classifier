import os
import sys
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
import pandas as pd

def train(train_iter, dev_iter, model, label_field, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    model.train()
    last_acc = 0
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                print('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))

            #if steps % args.save_interval == 0:
            #    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
            #    save_prefix = os.path.join(args.save_dir, 'snapshot')
            #    save_path = '{}_steps{}.pt'.format(save_prefix, steps)
            #    torch.save(model.state_dict(), save_path)

        # We evaluate the metric over the validation set epoch after epoch
        print('\nEpoch: ' + str(epoch), flush=True)  #Flush is needed to "real-time" printing in the Lovelace Cluster
        acc = eval(dev_iter, model, args)

        # We save the weights
        if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
        save_prefix = os.path.join(args.save_dir, 'snapshot')
        save_path = '{}_steps{}.pt'.format(save_prefix, steps)
        torch.save(model.state_dict(), save_path)

        if last_acc > acc and epoch > 15:
            print("Training stopped early due to validation set")
            print('Evaluation acc: {:.4f}% \n'.format(last_acc))                                                      
            
            C = compute_confusion_matrix(train_iter, model, args)
            print('Confusion matrix for the training set')
            print(C)
            print(C.sum(), (C[1,1] + C[2,2] + C[3,3] + C[4,4] + C[0,0])/C.sum())

            C = compute_confusion_matrix(dev_iter, model, args)
            print('Confusion matrix for the validation set')
            print(C)

            break

        last_acc = acc                        

def compute_confusion_matrix(dev_iter, model, args):
    model.eval()
    C = np.zeros([args.class_num, args.class_num])
    for batch in dev_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)
        if args.cuda:
                feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        C += confusion_matrix(target.data, torch.max(logit, 1)[1].view(target.size()).data, labels=range(args.class_num))

    model.train()
    return C

def eval(dev_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in dev_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1) 
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
        #print(torch.max(logit, 1)[1].view(target.size()).data)

    size = len(dev_iter.dataset)
    avg_loss = avg_loss/size
    accuracy = 100.0 * corrects/size

    #C = compute_confusion_matrix(dev_iter, model, args)
    #print(C)

    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy

def predict(data, model, text_field, label_field, cuda_flag, args):
    df = pd.read_excel(data)

    model.eval()
    labels = []
    probs = []

    df['pred'] = ''
    for i in range(args.class_num):
        df[label_field.vocab.itos[i+1]] = 0

    for i in range(len(df)):
        text = df.fallo[i]
        text = text_field.tokenize(text)
        text = text_field.preprocess(text)
        text = [[text_field.vocab.stoi[x] for x in text]]
        x = text_field.tensor_type(text)
        x = autograd.Variable(x, volatile=True)
        if cuda_flag:
            x = x.cuda()
        #print(x)
        output = model(x)

    # print(at)   # Print attention vectors
    #print(output.size())
        #print(F.softmax(output).data.cpu().numpy()[:args.class_num].shape)
        probs = F.softmax(output).data.cpu().numpy()[:args.class_num].tolist()
        for j in range(args.class_num):
            df.loc[i, label_field.vocab.itos[j+1]] = probs[j]

        _, predicted = torch.max(output, 0)
        labels.append(label_field.vocab.itos[predicted.data[0]+1])
    
    df['pred'] = labels
    print(df.head())
    df.to_csv('pred_proba.csv', float_format='%.5f')
    return labels
