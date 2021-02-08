def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    RMSE = 0
    with torch.no_grad():
        for i,data in enumerate(test_loader,1):
            data, target = data['field'], data['perm']
            data, target = data.to(device), target.to(device)
            # data, target, porosity, tort = data['field'], data['perm'], data['porosity'], data['tort']
            # data, target, porosity, tort = data.to(device), target.to(device), porosity.to(device), tort.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target,reduction='sum').item()
            if epoch == EPOCHS :
                test_data_set.extend(target.cuda().data.cpu().numpy())
                test_modeloutput_set.extend(output.cuda().data.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    R2_score = 1-test_loss/test_r2
    RMSE = pow(test_loss,0.5)
    test_loss_set.append(test_loss)
    RMSE_set.append(RMSE)
    R2_score_set.append(R2_score)

    print('\nTest set: Average loss: {:.6}, RMSE: {:.6}, R2_score: {:.6},Epoch:{}/{}\n'.format(test_loss, RMSE, R2_score, epoch, EPOCHS +1,
          ))