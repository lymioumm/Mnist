import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

dir = '/home/ZhangXueLiang/LiMiao/dataset/Mnist'


# 获取MNIST数据集
def get_data():
    # transformation = transforms.Compose([transforms.ToTensor(),
    #                                      transforms.Normalize((0.1307,),(0.3081))])
    transformation = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,),(0.3081))])
    train_dataset = datasets.MNIST(dir,train = True,transform = transformation,download = True)
    test_dataset = datasets.MNIST(dir,train = False,transform = transformation,download = True)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = 32,shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = 32,shuffle = True)




    sample_data = next(iter(train_loader))
    plot_img(sample_data[0][1])
    print(sample_data)
    # plot_img(sample_data[0][2])
    # for i in range(10):
    #     plt.figure()
    #     plt.imshow(train_loader.dataset.data[i].numpy())
    #     plt.show()
    #
    # x = torch.randn(2,2,2)
    # print(f'x.view(-1,1,4):{x.view(-1,1,8)}')
    # for (data,target) in train_loader:
    #     for i in range(4):
    #         plt.figure()
    #         print(target[1])
    #         plt.imshow(data[i].numpy()[0])
    #     break

    pass


def plot_img(image):
    image  =image.numpy()[0]
    mean = 0.1307
    std = 0.3081
    image = ((mean * image) + std)
    plt.imshow(image,cmap = 'gray')
    plt.show()
    plt.savefig('Mnist.jpg')      # 保存图片文件在服务器端
    pass

def main():
    # 获取MNIST数据集
    get_data()

    pass

if __name__ == '__main__':
    main()