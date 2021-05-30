# AI FREE 周末讀書會: PyTorch/ Image Classification 
> A step-by-step workthrough tutorial

> Eric

> 2021/05/29

# Specify Env:
- Python 3.7.10
- PyTorch 1.7.1 (cpu version) (gpu version works actually well)

# Outline:
- Step 01: Installation
- Step 02: Get Dataset
- Step 03: Arrange File Structure
- Step 04: Have a look of image
- Step 05: Define Custom Dataset Class
- Step 06: Define DataLoader w.r.t Dataset Class
- Step 07: Define LeNet5 - like Structure
- Step 08: Define Loss Function
- Step 09: Training Phase
- Step 10: Plot Accuracy & Loss Curves
- Step 11: Testing Phase

# Start:

- Step 01: Installation
    - Anaconda
    - Conda Create Env

        ```bash
        # 示範使用 CPU，請使用這個，Python 3.7 和 PyTorch 1.7.X 是好朋友 ^_^
        conda create --name "PT_CPU_1.7.1_DEMO" python=3.7

        # 確認新環境建置完成
        conda env list

        # 啟動新環境
        conda activate PT_CPU_1.7.1_DEMO
        ```

    - PyTorch: 1.7.X, and other useful packages (I spent 10 mins on this)

        ```python

        # 核心: 安裝 pytorch 1.7.1 CPU Version
        conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch

        ### Other Useful Packages
        # 1. GUI 介面，方便寫程式 → jupyter notebook
        conda install jupyter notebook

        # 2. 用來讀 Metadata.csv → pandas
        conda install pandas

        # 3. 知道 loop 的進度 → tqdm
        conda install tqdm

        # 4. 讀影像，其他選擇如 PIL，總之這邊使用 OpenCV → opencv
        conda install -c conda-forge opencv

        # 5. 畫圖 → matplotlib
        conda install matplotlib

        # 6. 矩陣運算 → numpy
        conda install numpy
        ```

    - 示範上考慮到大家不一定都有顯卡，所以使用 CPU，若想要裝 higher version (e.g. 1.8.X) or GPU Version，請參考以下網址。
    - PyTorch Installation Link: [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)
- Step 02: Get Dataset

    ### Temp: Download From Google Cloud

    - AIdea AOI Detection: [https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4)

        ![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step02_01.png)

    - Download Page:

        ![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step02_02.png)

- Step 03: Arrange File Structure
    - 預期的專案結構如下:
        - . / test_images/
        - . / train_images/
        - . / test.csv
        - . / train.csv
        - . / LeNet5.ipynb ← 這個 Python File 是要自己新建的

        ![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step03_01.png)

    - train_images

        ![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step03_02.png)

    - train.csv

        ![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step03_03.png)

---

- 在 PyTorch，Train 的時候提供 Batch 需要使用 DataLoader，而要使用 DataLoader 前，會需要先完成自定義的 Dataset Class。
- 下方正式進入程式碼，請大家移動 cd 至專案路徑當中，並開啟 jupyter notebook

```bash
jupyter notebook
```

然後點選: LeNet5.ipynb

![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step03_05.png)

---

- Step 04: Have a look of image
    - import 此步驟需要之套件

        ```python
        import pandas as pd
        import cv2 as cv
        import matplotlib.pyplot as plt
        ```

    - Mapping table: Python Dictionary is our friend!

        ```python
        label_map_table = {
            0: "normal",
            1: "void",
            2: "Horizontal Defect",
            3: "Vertical Defect",
            4: "Edge Defect",
            5: "Partical"
        }
        ```

    - Visualize it!

        ```python
        root_train = "./train_images/"
        root_test = "./test_images/"
        train_csv = "./train.csv"
        test_csv = "./test.csv"
        df_train = pd.read_csv(train_csv) # df stands for dataframe

        id = 137 # change this number to see other outcome
        png_img = cv.imread(root_train + df_train.ID[id])
        label = df_train.Label[id]
        print(f"[Label] => {df_train.Label[id]}; [Label Actually Means] => {label_map_table[label]}")
        plt.imshow(png_img)
        plt.show()
        ```

    - Conclusion:

        ![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step04_01.png)

- Step 05: Define Custom Dataset Class
    - import 此步驟需要的套件

        ```python
        from torch.utils.data.dataset import Dataset
        from torchvision import transforms
        import numpy as np
        import pandas as pd
        import cv2 as cv
        ```

    - 重頭戲: 動手寫 Custom DataLoader

        ```python
        # [Input Args]
        # 1. target_csv <string>: It's the metadata file describe the name of image and its label.
        # 2. root_path  <string>: It's the path to the image folder. Combination of this and name is the full path to the image.
        # 3. height <int>: Use this for elastically resize image to desired shape.
        # 4. width <int>: Use this for elastically resize image to desired shape.
        class AOI_Dataset(Dataset):
            
            # perform logic operation: think what kind of info I need when loading data
            def __init__(self, target_csv, root_path, height, width, transform = None):
                
                # height, width
                pass
                
                # attach static info to self 
                pass
                
                # 1. Read CSV file through root_path (REF: Step 04)
                pass
                
                # 2. Remember the length (by self) (REF: def __len__(self))
                pass
                
                # 3. transform (you can also attach a "Function" to self)
                pass
            
            # input: index
            # output: pair of (image, lable)
            def __getitem__(self, index):
                # Read images
                img = cv.imread(self.root_path + self.df.ID[index])
                
                # Use resize to a smaller shape when training takes so long.
                img_resize = cv.resize(img, (self.height, self.width))

                # To Tensor
                img_tensor = self.transforms(np.uint8(img_resize))
                
                # Get label
                label = self.df.Label[index]
                
                return (img_tensor, label)
            
            def __len__(self):
                return self.count
        ```

    - REF: PyTorch 官方 Dataset & DataLoader 教學 [[Link](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)]
- Step 06: Define DataLoader w.r.t Dataset Class
    - import 此步驟需要的套件

        ```python
        import torch
        from torchvision import transforms
        ```

    - DataLoader

        ```python
        height = 512
        width = 512

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # add mroe in the future
        ])

        Train_Dataset = AOI_Dataset(target_csv = train_csv, root_path = root_train, height = height, width = width, transform = transform_train)

        batch_size = 8

        Train_DataLoader = torch.utils.data.DataLoader(Train_Dataset, batch_size = batch_size)
        ```

- Step 07: Define LeNet5-like Structure
    - Yann Lecun

        ![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step07_01.png)

        - Original Version of LeNet5 in his [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).

        ![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step07_02.png)

    - Structure of Original LeNet5

        ![](https://github.com/Ratherman/AI/blob/main/DeepLearning/HW3/imgs/LeNet_Mnist.png)

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    class LeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d( 3,  6, 3, padding = 1) # 加深 channel
            self.conv2 = nn.Conv2d( 6, 16, 3, padding = 1) # 加深 channel
            self.conv3 = nn.Conv2d(16, 50, 3, padding = 1) # 加深 channel
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(50 * 64 * 64, 120) # Why 64 Why 50
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 6)
            
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x))) # 512 x 512 x  3 -> 256 x 256 x  6
            x = self.pool(F.relu(self.conv2(x))) # 256 x 256 x  6 -> 128 x 128 x 16
            x = self.pool(F.relu(self.conv3(x))) # 128 x 128 x 16 ->  64 x  64 x 50
            x = x.view(-1, 50 * 64 * 64)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            #x = F.softmax(x)
            return x
        
    lenet = LeNet()
    print(lenet.to(device))
    ```

    - API lookup [[Link](https://pytorch.org/docs/stable/index.html)]
        - Conv2D [[Link](https://pytorch.org/docs/1.7.1/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)]

            ![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step07_03.png)

- Step 08: Define Loss Function and setup Hyper Parameters

    ```python
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(lenet.parameters(), lr=1e-4) # learning rate is also adjustable
    epoch = 10 # change this to whatever number you'd like
    ```

- Step 09: Training Phase

    ```python
    from tqdm import tqdm
    import time

    tic = time.time()
    train_acc_list = []
    val_acc_list = []
    loss_list = []
    print_probe_num = 10

    for epoch in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        
        for i, data in enumerate(Train_DataLoader, 0):
            # Select input and output pair
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Clear gradient
            optimizer.zero_grad()

            # Forward Propagation
            outputs = lenet(inputs.float())
            
            # Compute Loss
            loss = criterion(outputs, labels)
            
            # Backward Propagation
            loss.backward()
            
            # Update Weight
            optimizer.step()

            # Just want to calculate the running loss 移動平均 loss!!
            running_loss += loss.item()
            if i % print_probe_num == (print_probe_num - 1):
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_probe_num))
                loss_list.append(running_loss / print_probe_num)
                running_loss = 0.0
                
        correct = 0
        total = 0
        
        # Train
        with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
            for datum in tqdm(Train_DataLoader):

                imgs, labs = datum[0].to(device), datum[1].to(device)
                # calculate outputs by running images through the network 
                outputs = lenet(imgs.float())
                # the class with the highest energy is what we choose as prediction
                _, preds = torch.max(outputs.data, 1)
                
                total += labs.size(0)
                correct += (preds == labs).sum().item()
            train_acc_list.append(float(correct)/float(total))
            print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))

            
    toc = time.time()
    print(f"Spend {round(toc - tic, 2)} (sec)")
    print('Finished Training')
    ```

    ![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step09_00.png)

    - Why zero_grad
        - What step(), backward(), and zero_grad() do [[Link](https://discuss.pytorch.org/t/what-step-backward-and-zero-grad-do/33301)]

            ![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step09_01.png)

- Step 10: Plot Accuracy & Loss Curves

    ```python
    import matplotlib.pyplot as plt

    ## Accuracy

    plt.figure(figsize = (20, 10))
    plt.title("LeNet5: Accuracy Curve", fontsize = 24)
    plt.xlabel("Epochs"    , fontsize = 20)
    plt.ylabel("Accuracy %", fontsize = 20)
    plt.plot(train_acc_list, label = "train acc.")
    plt.legend(loc = 2, fontsize = 20)
    plt.show()

    ## Loss

    plt.figure(figsize=(20, 10))
    plt.title("LeNet5: Loss Curve", fontsize = 24)
    plt.plot(loss_list)
    plt.xlabel("Probes", fontsize = 20)
    plt.ylabel("Loss", fontsize = 20)
    plt.show()
    ```

    ![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step10_01.png)

    ![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step10_02.png)

- Step 11: Testing Phase

    ```python
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    Test_Dataset = AOI_Dataset(target_csv = test_csv, root_path = root_test, height = height, width = width, transform = transform_test)
    Test_DataLoader = torch.utils.data.DataLoader(dataset = Test_Dataset, batch_size = 1, shuffle = False)
    Name_of_csv_file = "AI.FREE.SUCCESS.csv"

    df_test = pd.read_csv(test_csv)
    df_test_np = df_test.to_numpy()

    count = -1
    with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
        for datum in tqdm(Test_DataLoader):
            count = count + 1
            imgs = datum[0].to(device)
            # calculate outputs by running images through the network 
            outputs = lenet(imgs.float())
            # the class with the highest energy is what we choose as prediction
            _, preds = torch.max(outputs.data, 1)
            df_test_np[count][1] = float(preds)
            
    df = pd.DataFrame(df_test_np, columns = ['ID','Label'])
    df.to_csv(Name_of_csv_file, index=False)
    ```

    ![](https://github.com/Ratherman/AI/blob/main/My_Tutorial/20210529_PyTorch_Image_Classification/imgs/step11_01.png)

# Future Plan

- Submit the result to AIdea
- Validation Dataset
- Data Augmentation
- Train on GPU
- Estimate Usage of GPU Memory