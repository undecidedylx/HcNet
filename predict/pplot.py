# -- coding: utf-8 --

df = pd.read_csv(
    r'/remote-home/hongzhangxin/pytorch_project/TB_Classfiy/result/self_weight/loss_acc_EfficientNet_3D.csv')
losses = df["loss"].values
accuracies = df["acc"].values
plt.title("DenseNet_3D_acc_loss")
plt.plot(losses, label="Loss")
plt.plot(accuracies, label="Acc")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()

plt.savefig("/remote-home/hongzhangxin/pytorch_project/TB_Classfiy/reslut_img/acc_loss_EfficientNet_3D.png")
plt.show()
plt.close()
