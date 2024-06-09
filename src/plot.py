import matplotlib.pyplot as plt
import pandas as pd

#plotting metrics from lightning logs

df = pd.read_csv('lightning_logs/version_254/metrics.csv')


# plt.plot(df['epoch'], df['g_adv_loss'], label='Generator adversarial loss')
# plt.plot(df['epoch'], df['d_clear_loss_val'], label='Clear adversarial loss')
# plt.plot(df['epoch'], df['d_noisy_loss_val'], label='Noisy adversarial loss')
# plt.plot(df['epoch'], df['d_adv_loss'], label='Discriminator adversarial loss')
# plt.plot(df['epoch'], df['snr'], label='SNR')
plt.plot(df['epoch'], df['g_rec_loss'], label='Generator reconstruction loss')
plt.legend()
plt.xlabel('epochs')
plt.show()
