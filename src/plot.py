import matplotlib.pyplot as plt
import pandas as pd

#plotting metrics from lightning logs

df = pd.read_csv('lightning_logs/version_167/metrics.csv')


plt.plot(df['epoch'], df['g_adv_loss'], label='g_adv_loss')
plt.plot(df['epoch'], df['d_clear_loss_val'], label='clear_loss')
plt.plot(df['epoch'], df['d_noisy_loss_val'], label='noisy_loss')
plt.plot(df['epoch'], df['d_adv_loss'], label='d_adv_loss')
plt.plot(df['epoch'], df['snr'], label='snr')
plt.plot(df['epoch'], df['g_rec_loss'], label='g_rec_loss')
plt.legend()

plt.show()
