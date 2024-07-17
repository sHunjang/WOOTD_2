import matplotlib.pyplot as plt

# 데이터
data = [69, 21, 26, 35]

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.bar(range(len(data)), data, color='green', edgecolor='none')
plt.title('Epoch 40 Model')
plt.xlabel('Combination')
plt.ylabel('Similairty')
plt.legend()
plt.grid(False)
plt.xticks(range(len(data)))
plt.show()
