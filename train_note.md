# 2022-06-20训练日志

使用README.md上的微调的训练方法，训练了head2的数据，生成的head2-20220616.zip的包，生成的数据还可以，只是无法区分男女
而且发现一个问题，那就是如果没有匹配到合适的图像的话，就会出现只用原始数据的情况，而不是随机生成一个图像。
而且，预训练过的数据上是以欧美人为标准的，导致风格上有所偏向欧美。

使用README.md上的微调的训练方法，训练了simpsons的数据，得到simpsons-20220616-fail.zip，如果只训练destylize.py的部分，发现只是修改皮肤的颜色和头发的颜色，
其他的纹理全都没有学习进去。但是如果把后续的微调步骤也都执行了的话，会发现学习到的东西完全不对，为了找到是为了所以执行了下面的步骤。

使用README.md上的训练方法，但是没有DualStyleGAN上的预训练数据，而是自己训练了simpsons的数据，数据保存在simpsons-20220620.zip，有得到效果，但是效果非常的不理想，颜色和基本的样子都学进去了，但是学的不够准确，脸型是有严重的偏差的，需要在后续的步骤中找寻原因并修正。

接下来，需要实验的是，是否是因为训练的数据太多导致的，或者是因为StyleGAN的预训练的部分也是需要调整的。

# 2022-06-21

使用README.md上的训练方法，训练的caricature，只用了2张的数据，训练的数据非常的差，无法看到任何特征，也就是说，样本数量过少是行不通的，数据保存在caricature-20220621.zip下面

# 2022-06-23

使用README.md上的训练方法，训练的simpsons，再训练finetune_dualstylegan.py的时候，依次实验了perc_loss的参数1、0.5、0.25，效果都不太好，还实验了style_loss是0.2的情况，也不太理想，学到了很多的眼睛

接下来需要认真再把模型结构再认真看一遍，找明白是为什么。

# 2022-06-27

开始读代码
在``style_transfer.py``最后生成的方法中，
首先从``generator-001500.pt``中加载出``g_ema``
然后加载从``Pixel2style2pixel``中生成的``encoder.pt``，来处理latent code
加载``exstyle_code.npy``中的外部风格，该文件是来自于``destylize.py``中处理过后的style code z^+_e的集合
然后经过``encoder``的处理之后，从``DualStyleGAN``中的``generator``方法生成想要的结果，然后保存成图片

以上出现的问题，分别使用到了``Pixel2style2pixel``和``StyleGAN``生成的网络结构和结果来处理
重点是里面的提到的latent code到底是什么，是需要深入学习和了解的

然后还看了``DualStyleGAN``中的网络，主要牵涉到的也是``StyleGAN``，所以这一块也是需要深入研读的。

以下来自于https://zhuanlan.zhihu.com/p/369946876
流行学习 https://www.zhihu.com/question/24015486
## latent code
“潜在空间”（Latent Space）的概念很重要，因为它的用途是“深度学习”的核心-学习数据的特征并简化数据表示形式以寻找模式。

# 2022-06-28
https://zhuanlan.zhihu.com/p/67822290
z隐藏空间，是只如果原来不是线性可分的，通过`特征转换`可以变成一个线性可分的空间






