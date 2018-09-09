2018全国高校大数据应用创新大赛（技能赛）
===
赛题地址http://117.50.29.62/pc/competition_topic.jsp

***思路***：对训练数据，每条样本进行全排列产生120种结果，其标签都一样；其他的操作编码，shuffle，去重等，之后带入神经网络模型训练。决赛中一个新思路（预赛时每想到，如果用这思路是很容易做到1的），对部分测试样本进行全排列，预测其120种结果，然后用众数或knn的思路取最终标签。

预赛：
---

>preliminary<br>
>>stacking.py<br>
>>nn_model.py<br>
>>pre.py

决赛：
---

>semifinal<br>
>>utils.py<br>
>>build_features.py<br>
>>stacking.py<br>
>>pre.py
