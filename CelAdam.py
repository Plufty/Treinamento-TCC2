


import os
import torch
import torchvision
import time
from torch import nn, optim
from torchvision import datasets, transforms, models
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet


def CelAdam (learningRate, data_dir):


    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    # # Carregar Pasta 

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}


    # # Carregando o modelo pré treinado 

    #Definindo Número de Classes
    num_classes = 2


    # ## AlexNet
    model_alex = models.alexnet(pretrained=True)
    num_ftrs_alex = model_alex.classifier[6].in_features
    model_alex.classifier[6] = nn.Linear(num_ftrs_alex, num_classes)


    # ## EfficientNet
    model_efficient = EfficientNet.from_pretrained('efficientnet-b0')
    num_ftrs_efficient = model_efficient._fc.in_features
    model_efficient._fc = nn.Linear(num_ftrs_efficient, num_classes)


    # ## ResNet50
    model_res = models.resnet50(pretrained=True)
    num_ftrs_res = model_res.fc.in_features
    model_res.fc = nn.Linear(num_ftrs_res, num_classes)


    # ## ShuffleNet
    model_shuffle= models.shufflenet_v2_x1_0(pretrained=True)
    num_ftrs_shuffle = model_shuffle.fc.in_features
    model_shuffle.fc = nn.Linear(num_ftrs_shuffle, num_classes)


    # ## SqueezeNet
    model_squeeze= models.squeezenet1_0(pretrained=True)
    model_squeeze.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model_squeeze.num_classes = num_classes


    # ## VGG16
    model_vgg= models.vgg16(pretrained=True)
    num_ftrs_vgg = model_vgg.classifier[6].in_features
    model_vgg.classifier[6] = nn.Linear(num_ftrs_vgg, num_classes)


    # # Verificar o device do treinamento
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device

    # # Movendo o modelo para a GPU
    model_alex.to(device)
    model_efficient.to(device)
    model_res.to(device)
    model_shuffle.to(device)
    model_squeeze.to(device)
    model_vgg.to(device)


    # # Loop de treinamento

    # # AlexNet

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model_alex.parameters(), lr=learningRate)


    num_epochs = 30
    perdas_alex = list()
    perdas_alex.append('model,criterion,optimizer,epoch,epoch_time,train_loss,val_loss,train_accuracy,val_accuracy\n')
    ini = time.time()

    for epoch in tqdm(range(num_epochs)):
        epoch_ini = time.time()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Loop de treinamento
        for i, (inputs, labels) in enumerate(dataloaders['train'], 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            inputs = nn.functional.interpolate(inputs, size=(224, 224))

            optimizer.zero_grad()

            outputs = model_alex(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            # Calculando acurácia do treinamento
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        # Loop de validação
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for j, (inputs_val, labels_val) in enumerate(dataloaders['val'], 0):
                inputs_val = inputs_val.to(device)
                labels_val = labels_val.to(device)
                
                inputs_val = nn.functional.interpolate(inputs_val, size=(224, 224))

                outputs_val = model_alex(inputs_val)
                val_loss += criterion(outputs_val, labels_val).item()

                # Calculando acurácia da validação
                _, predicted_val = torch.max(outputs_val.data, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted_val == labels_val).sum().item()

        # Imprimindo resultados
        epoch_fim = time.time()
        train_loss = running_loss / len(dataloaders['train'])
        train_accuracy = 100 * correct_train / total_train
        val_loss /= len(dataloaders['val'])
        val_accuracy = 100 * correct_val / total_val

        print('Epoch [%d], epoch time: %.3f s, train loss: %.3f, val loss: %.3f, train accuracy: %.3f %%, val accuracy: %.3f %%' % (epoch+1, epoch_fim-epoch_ini, train_loss, val_loss, train_accuracy, val_accuracy))
        perdas_alex.append('AlexNet,Cross_Entropy_Loss,Adam,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n' % (epoch+1, epoch_fim-epoch_ini, train_loss, val_loss, train_accuracy, val_accuracy))

    fim = time.time() 
    tempo = fim-ini

    print('Time [%.3f] seconds or [%.3f] minutes' % (tempo, tempo/60))
    perdas_alex.append('Time [%.3f] seconds or [%.3f] minutes\n' % (tempo, tempo/60))


    # # EfficientNet
    optimizer = optim.Adam(model_efficient.parameters(), lr=learningRate)

    num_epochs = 30
    perdas_efficient = list()
    perdas_efficient.append('model,criterion,optimizer,epoch,epoch_time,train_loss,val_loss,train_accuracy,val_accuracy\n')
    ini = time.time()

    for epoch in tqdm(range(num_epochs)):
        epoch_ini = time.time()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Loop de treinamento
        for i, (inputs, labels) in enumerate(dataloaders['train'], 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model_efficient(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculando acurácia do treinamento
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        # Loop de validação
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for j, (inputs_val, labels_val) in enumerate(dataloaders['val'], 0):
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)

                outputs_val = model_efficient(inputs_val)
                val_loss += criterion(outputs_val, labels_val).item()

                # Calculando acurácia da validação
                _, predicted_val = torch.max(outputs_val.data, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted_val == labels_val).sum().item()

        # Imprimindo resultados
        epoch_fim = time.time()
        train_loss = running_loss / len(dataloaders['train'])
        train_accuracy = 100 * correct_train / total_train
        val_loss /= len(dataloaders['val'])
        val_accuracy = 100 * correct_val / total_val

        print('Epoch [%d], epoch time: %.3f s, train loss: %.3f, val loss: %.3f, train accuracy: %.3f %%, val accuracy: %.3f %%' % (epoch+1, epoch_fim-epoch_ini, train_loss, val_loss, train_accuracy, val_accuracy))
        perdas_efficient.append('EfficientNet,Cross_Entropy_Loss,Adam,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n' % (epoch+1, epoch_fim-epoch_ini, train_loss, val_loss, train_accuracy, val_accuracy))

    fim = time.time() 
    tempo = fim-ini

    print('Time [%.3f] seconds or [%.3f] minutes' % (tempo, tempo/60))
    perdas_efficient.append('Time [%.3f] seconds or [%.3f] minutes\n' % (tempo, tempo/60))


    # # ResNet50
    optimizer = optim.Adam(model_res.parameters(), lr=learningRate)

    num_epochs = 30
    perdas_res = list()
    perdas_res.append('model,criterion,optimizer,epoch,epoch_time,train_loss,val_loss,train_accuracy,val_accuracy\n')
    ini = time.time()

    for epoch in tqdm(range(num_epochs)):
        epoch_ini = time.time()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Loop de treinamento
        for i, (inputs, labels) in enumerate(dataloaders['train'], 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            inputs = nn.functional.interpolate(inputs, size=(224, 224))

            optimizer.zero_grad()

            outputs = model_res(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            # Calculando acurácia do treinamento
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        # Loop de validação
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for j, (inputs_val, labels_val) in enumerate(dataloaders['val'], 0):
                inputs_val = inputs_val.to(device)
                labels_val = labels_val.to(device)
                
                inputs_val = nn.functional.interpolate(inputs_val, size=(224, 224))

                outputs_val = model_res(inputs_val)
                val_loss += criterion(outputs_val, labels_val).item()

                # Calculando acurácia da validação
                _, predicted_val = torch.max(outputs_val.data, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted_val == labels_val).sum().item()

        # Imprimindo resultados
        epoch_fim = time.time()
        train_loss = running_loss / len(dataloaders['train'])
        train_accuracy = 100 * correct_train / total_train
        val_loss /= len(dataloaders['val'])
        val_accuracy = 100 * correct_val / total_val

        print('Epoch [%d], epoch time: %.3f s, train loss: %.3f, val loss: %.3f, train accuracy: %.3f %%, val accuracy: %.3f %%' % (epoch+1, epoch_fim-epoch_ini, train_loss, val_loss, train_accuracy, val_accuracy))
        perdas_res.append('ResNet50,Cross_Entropy_Loss,Adam,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n' % (epoch+1, epoch_fim-epoch_ini, train_loss, val_loss, train_accuracy, val_accuracy))

    fim = time.time() 
    tempo = fim-ini

    print('Time [%.3f] seconds or [%.3f] minutes' % (tempo, tempo/60))
    perdas_res.append('Time [%.3f] seconds or [%.3f] minutes\n' % (tempo, tempo/60))


    # # ShuffleNet
    optimizer = optim.Adam(model_shuffle.parameters(), lr=learningRate)

    num_epochs = 30
    perdas_shuffle = list()
    perdas_shuffle.append('model,criterion,optimizer,epoch,epoch_time,train_loss,val_loss,train_accuracy,val_accuracy\n')
    ini = time.time()

    for epoch in tqdm(range(num_epochs)):
        epoch_ini = time.time()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Loop de treinamento
        for i, (inputs, labels) in enumerate(dataloaders['train'], 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            inputs = nn.functional.interpolate(inputs, size=(224, 224))

            optimizer.zero_grad()

            outputs = model_shuffle(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            # Calculando acurácia do treinamento
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        # Loop de validação
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for j, (inputs_val, labels_val) in enumerate(dataloaders['val'], 0):
                inputs_val = inputs_val.to(device)
                labels_val = labels_val.to(device)
                
                inputs_val = nn.functional.interpolate(inputs_val, size=(224, 224))

                outputs_val = model_shuffle(inputs_val)
                val_loss += criterion(outputs_val, labels_val).item()

                # Calculando acurácia da validação
                _, predicted_val = torch.max(outputs_val.data, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted_val == labels_val).sum().item()

        # Imprimindo resultados
        epoch_fim = time.time()
        train_loss = running_loss / len(dataloaders['train'])
        train_accuracy = 100 * correct_train / total_train
        val_loss /= len(dataloaders['val'])
        val_accuracy = 100 * correct_val / total_val

        print('Epoch [%d], epoch time: %.3f s, train loss: %.3f, val loss: %.3f, train accuracy: %.3f %%, val accuracy: %.3f %%' % (epoch+1, epoch_fim-epoch_ini, train_loss, val_loss, train_accuracy, val_accuracy))
        perdas_shuffle.append('ShuffleNet,Cross_Entropy_Loss,Adam,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n' % (epoch+1, epoch_fim-epoch_ini, train_loss, val_loss, train_accuracy, val_accuracy))

    fim = time.time() 
    tempo = fim-ini

    print('Time [%.3f] seconds or [%.3f] minutes' % (tempo, tempo/60))
    perdas_shuffle.append('Time [%.3f] seconds or [%.3f] minutes\n' % (tempo, tempo/60))


    # SqueezeNet
    optimizer = optim.Adam(model_squeeze.parameters(), lr=learningRate)

    num_epochs = 30
    perdas_squeeze = list()
    perdas_squeeze.append('model,criterion,optimizer,epoch,epoch_time,train_loss,val_loss,train_accuracy,val_accuracy\n')
    ini = time.time()

    for epoch in tqdm(range(num_epochs)):
        epoch_ini = time.time()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Loop de treinamento
        for i, (inputs, labels) in enumerate(dataloaders['train'], 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            inputs = nn.functional.interpolate(inputs, size=(224, 224))

            optimizer.zero_grad()

            outputs = model_squeeze(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            # Calculando acurácia do treinamento
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        # Loop de validação
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for j, (inputs_val, labels_val) in enumerate(dataloaders['val'], 0):
                inputs_val = inputs_val.to(device)
                labels_val = labels_val.to(device)
                
                inputs_val = nn.functional.interpolate(inputs_val, size=(224, 224))

                outputs_val = model_squeeze(inputs_val)
                val_loss += criterion(outputs_val, labels_val).item()

                # Calculando acurácia da validação
                _, predicted_val = torch.max(outputs_val.data, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted_val == labels_val).sum().item()

        # Imprimindo resultados
        epoch_fim = time.time()
        train_loss = running_loss / len(dataloaders['train'])
        train_accuracy = 100 * correct_train / total_train
        val_loss /= len(dataloaders['val'])
        val_accuracy = 100 * correct_val / total_val

        print('Epoch [%d], epoch time: %.3f s, train loss: %.3f, val loss: %.3f, train accuracy: %.3f %%, val accuracy: %.3f %%' % (epoch+1, epoch_fim-epoch_ini, train_loss, val_loss, train_accuracy, val_accuracy))
        perdas_squeeze.append('SqueezeNet,Cross_Entropy_Loss,Adam,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n' % (epoch+1, epoch_fim-epoch_ini, train_loss, val_loss, train_accuracy, val_accuracy))

    fim = time.time() 
    tempo = fim-ini

    print('Time [%.3f] seconds or [%.3f] minutes' % (tempo, tempo/60))
    perdas_squeeze.append('Time [%.3f] seconds or [%.3f] minutes\n' % (tempo, tempo/60))


    # VGG16
    optimizer = optim.Adam(model_vgg.parameters(), lr=learningRate)

    num_epochs = 30
    perdas_vgg = list()
    perdas_vgg.append('model,criterion,optimizer,epoch,epoch_time,train_loss,val_loss,train_accuracy,val_accuracy\n')
    ini = time.time()

    for epoch in tqdm(range(num_epochs)):
        epoch_ini = time.time()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Loop de treinamento
        for i, (inputs, labels) in enumerate(dataloaders['train'], 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            inputs = nn.functional.interpolate(inputs, size=(224, 224))

            optimizer.zero_grad()

            outputs = model_vgg(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            # Calculando acurácia do treinamento
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        # Loop de validação
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for j, (inputs_val, labels_val) in enumerate(dataloaders['val'], 0):
                inputs_val = inputs_val.to(device)
                labels_val = labels_val.to(device)
                
                inputs_val = nn.functional.interpolate(inputs_val, size=(224, 224))

                outputs_val = model_vgg(inputs_val)
                val_loss += criterion(outputs_val, labels_val).item()

                # Calculando acurácia da validação
                _, predicted_val = torch.max(outputs_val.data, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted_val == labels_val).sum().item()

        # Imprimindo resultados
        epoch_fim = time.time()
        train_loss = running_loss / len(dataloaders['train'])
        train_accuracy = 100 * correct_train / total_train
        val_loss /= len(dataloaders['val'])
        val_accuracy = 100 * correct_val / total_val

        print('Epoch [%d], epoch time: %.3f s, train loss: %.3f, val loss: %.3f, train accuracy: %.3f %%, val accuracy: %.3f %%' % (epoch+1, epoch_fim-epoch_ini, train_loss, val_loss, train_accuracy, val_accuracy))
        perdas_vgg.append('VGG16,Cross_Entropy_Loss,Adam,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n' % (epoch+1, epoch_fim-epoch_ini, train_loss, val_loss, train_accuracy, val_accuracy))

    fim = time.time() 
    tempo = fim-ini

    print('Time [%.3f] seconds or [%.3f] minutes' % (tempo, tempo/60))
    perdas_vgg.append('Time [%.3f] seconds or [%.3f] minutes\n' % (tempo, tempo/60))


    torch.save(model_alex.state_dict(), data_dir+'\modelo_treinado_Cel_Adam_alex_learningRate{}.pth'.format(learningRate))
    print("Modelo_alex treinado salvo com sucesso!")
    arquivo = open(data_dir+'\perdas_alex_Cel_Adam_learningRate{}.csv'.format(learningRate),'w')
    for epoch in tqdm(range(num_epochs+2)):
        arquivo.write(perdas_alex[epoch])
    arquivo.close()

    torch.save(model_efficient.state_dict(), data_dir+'\modelo_treinado_Cel_Adam_efficient_learningRate{}.pth'.format(learningRate))
    print("Modelo_efficient treinado salvo com sucesso!")
    arquivo = open(data_dir+'\perdas_efficient_Cel_Adam_learningRate{}.csv'.format(learningRate),'w')
    for epoch in tqdm(range(num_epochs+2)):
        arquivo.write(perdas_efficient[epoch])
    arquivo.close()

    torch.save(model_res.state_dict(), data_dir+'\modelo_treinado_Cel_Adam_res_learningRate{}.pth'.format(learningRate))
    print("Modelo_res treinado salvo com sucesso!")
    arquivo = open(data_dir+'\perdas_res_Cel_Adam_learningRate{}.csv'.format(learningRate),'w')
    for epoch in tqdm(range(num_epochs+2)):
        arquivo.write(perdas_res[epoch])
    arquivo.close()


    torch.save(model_shuffle.state_dict(), data_dir+'\modelo_treinado_Cel_Adam_shuffle_learningRate{}.pth'.format(learningRate))
    print("Modelo_shuffle treinado salvo com sucesso!")
    arquivo = open(data_dir+'\perdas_shuffle_Cel_Adam_learningRate{}.csv'.format(learningRate),'w')
    for epoch in tqdm(range(num_epochs+2)):
        arquivo.write(perdas_shuffle[epoch])
    arquivo.close()


    torch.save(model_squeeze.state_dict(), data_dir+'\modelo_treinado_Cel_Adam_squeeze_learningRate{}.pth'.format(learningRate))
    print("Modelo_squeeze treinado salvo com sucesso!")
    arquivo = open(data_dir+'\perdas_squeeze_Cel_Adam_learningRate{}.csv'.format(learningRate),'w')
    for epoch in tqdm(range(num_epochs+2)):
        arquivo.write(perdas_squeeze[epoch])
    arquivo.close()


    torch.save(model_vgg.state_dict(), data_dir+'\modelo_treinado_Cel_Adam_vgg_learningRate{}.pth'.format(learningRate))
    print("Modelo_vgg treinado salvo com sucesso!")
    arquivo = open(data_dir+'\perdas_vgg_Cel_Adam_learningRate{}.csv'.format(learningRate),'w')
    for epoch in tqdm(range(num_epochs+2)):
        arquivo.write(perdas_vgg[epoch])
    arquivo.close()
