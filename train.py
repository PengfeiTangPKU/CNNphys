def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_sum = 0
    for i,data in enumerate(train_loader,1):
        data, target = data['field'], data['perm']
        data, target = data.to(device), target.to(device)
        # data, target, porosity, tort = data['field'], data['perm'], data['porosity'], data['tort']
        # data, target, porosity, tort = data.to(device), target.to(device), porosity.to(device), tort.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output,target)
        # if weight_decay >0 :
        #     loss = loss + reg_loss(model)
        train_loss = loss.item()
        train_loss_set.append(train_loss)
        loss_sum += train_loss
        if epoch == EPOCHS :
            train_data_set.extend(target.cuda().data.cpu().numpy())
            train_modeloutput_set.extend(output.cuda().data.cpu().numpy())
        if(i+1)%30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}\t'.format(
                epoch,i * len(data), len(train_loader.dataset),100. *i / len(train_loader), loss.item()))
        loss.backward()
        optimizer.step()
    loss_aver = loss_sum/(len(train_loader.dataset)/BATCH_SIZE)
    R2_score_train = 1-loss_aver/train_r2
    R2_score_train_set.append(R2_score_train)