import tensorflow as tf
file1 = open("16*1/metrics/metrics_text1","w+")#
avgs=0
avgp=0
avgm=0
sess=tf.InteractiveSession()
for i in range(0,100):
    ssim1 = tf.image.ssim(tf.convert_to_tensor(x_test1[i]),tf.convert_to_tensor(ans1[i]),max_val=1).eval()
    file1.write(str(i)+". = \t")
    file1.write("SSIM:\t"+str(ssim1))
    print(ssim1)
    print(np.amax(ans1[i]))
    t=tf.image.psnr(tf.convert_to_tensor(x_test1[i]),tf.convert_to_tensor(ans1[i]),max_val=1).eval()
    file1.write("\tPSNR:\t"+str(t))
    t1=tf.image.ssim_multiscale(tf.convert_to_tensor(x_test1[i]),tf.convert_to_tensor(ans1[i]),max_val=1).eval()
    file1.write("\tMs-ssim:\t"+str(t1)+"\n")
    avgs+=ssim1
    avgm+=t1
    print(t)
    print(t1)
    avgp+=t
    print(i)
file1.write("AVERAGE. = \t")
file1.write("SSIM:\t"+str(avgs/100))
file1.write("MS-SSIM:\t"+str(avgm/100))
file1.write("PSNR:\t"+str(avgp/100)+"\n")
print("avg=",avgs/100,avgp/100,avgm/100)
file1.close()
sess.close()
