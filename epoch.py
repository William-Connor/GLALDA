# import re
# import pandas as pd
#
# # 读取epoch.txt文件
# with open('epoch.txt') as f:
#     text = f.read()
#
# # 使用正则表达式匹配train_loss值
# pattern = r'the \d+ times train_loss is (\d+\.\d+)'
# train_losses = re.findall(pattern, text)
#
# # 将train_loss值转换成float类型保存到列表
# train_losses = [float(loss) for loss in train_losses]
#
# # 保存到excel文件
# df = pd.DataFrame({'train_loss': train_losses})
# df.to_excel('train_losses.xlsx', index=False)
import pandas as pd
import matplotlib.pyplot as plt

THRESHOLD = 0.01

df = pd.read_excel('train_losses.xlsx')

for i in range(1,len(df)):
    if abs(df.loc[i,'train_loss'] - df.loc[i-1,'train_loss']) < THRESHOLD:
        print(f'Epoch {i} loss did not decrease significantly')

plt.plot(df['train_loss'])
plt.savefig('loss_curve.png')

