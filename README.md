# machine-learning-homework-7-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning Homework 7 Solved](https://www.ankitcodinghub.com/product/machine-learning-solved-8/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;121240&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning Homework 7 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
Machine Learning HW7

BERT – Question Answering

Outline

1. Task Introduction

2. Tutorial 3. Hints

4. Grading

5. Report

6. Regulations

1. Task Introduction

Task: Extractive Question Answering

https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/bert_v8.pdf#page=23

Dataset: DRCD &amp; ODSQA

DRCD: 台達閱讀理解資料集 Delta Reading Comprehension Dataset

ODSQA: Open-Domain Spoken Question Answering Dataset

● train: DRCD + DRCD-TTS

○ 10524 paragraphs, 31690 questions

● dev: DRCD + DRCD-TTS

○ 1490 paragraphs, 4131 questions

● test: DRCD + ODSQA

○ 1586 paragraphs, 4957 questions

Dataset: Chinese Reading Comprehension

Dataset: Chinese Reading Comprehension

Paragraph

新加坡、馬來西亞的華文學術界在 1970年代後開始統一使用簡體中文；然而繁體字在媒體中普遍存在著，例如華人商店的招牌、舊告示及許多非學術類中文書籍，香港和臺灣所出版的書籍也有在市場上流動。當地許多中文報章都會使用「標題繁體字，內容簡化字」的方式讓簡繁中文並存。除此之外，馬來西亞所有本地中文報章之官方網站都是以繁體中文為主要文字。

Question: 新加坡的華文學術界在哪個年代後開始使用簡體中文 ?

Answer: 1970

Question: 馬來西亞的華人商店招牌主要使用什麼文字 ?

Answer: 繁體字

2. Tutorial

Tokenization

‘李宏毅教授2022機器學習’

tokenize

tokens_to_ids

[3330, 2131, 3675, 3136, 2956, 10550, 3582, 1690, 2119, 5424]

Tokenization

input_ids:

token_type_ids: 0 0 0 0 1 1 1 1 0 0

attention_mask: 1 1 1 1 1 1 1 1 0 0

Why Long Paragraph is an Issue?

Total sequence length = question length + paragraph length + 3 (special tokens) Maximum input sequence length of BERT is restricted to 512, why?

Self-Attention in transformer has complexity

What can we do?

Training

We know where the answer is in training!

Assumption: Info needed to answer the question can be found near the answer!

Simple solution: Just draw a window (as large as possible) around the answer!

e.g. window size = max_paragraph_len = 32

新加坡、馬來西亞的華文學術界在 1970年代後開始統一使用簡體中文；然而繁體字在媒體中普遍存在著，例如華人商店的招牌、舊告示及許多非學術類中文書籍，香港和臺灣所出版的書籍也有在市場上流動 …

Q: 新加坡的華文學術界在哪個年代後開始使用簡體中文 ? A: 1970

Q: 馬來西亞的華人商店招牌主要使用什麼文字 ? A: 繁體字

Testing

We do not know where the answer is in testing split into windows!

e.g. window size = max_paragraph_len = 32

新加坡、馬來西亞的華文學術界在 1970年代後開始統一使用簡體中文；然而繁體字在媒體中普遍存在著，例如華人商店的招牌、舊告示及許多非學術類中文書籍 ……

Q: 新加坡的華文學術界在哪個年代後開始使用簡體中文 ? A: 1970

Q: 馬來西亞的華人商店招牌主要使用什麼文字 ? A: 繁體字

For each window, model predicts a start score and an end score take the maximum to be answer!

start score start position end score end position total score

window 1 0.5 23 0.4 26 0.9

window 2 0.3 35 0.7 37 1.0

Answer:

position 35 to 37

3. Hints

Hints for beating baselines

❖ Simple:

➢ Sample code ❖ Medium:

➢ Apply linear learning rate decay

➢ Change value of “doc_stride”

❖ Strong:

➢ Improve preprocessing ➢ Try other pretrained models

❖ Boss:

➢ Improve postprocessing

➢ Further improve the above hints

Estimated training time

K80 T4 T4 (FP16) P100 V100

Simple 40m 20m 8m 10m 7m

Medium 40m 20m 8m 10m 7m

Strong 2h 1h 25m 35m 20m

Boss 12.5h 6h 2.5h 4.5h 2h

❖ Training Tips (Optional):

➢ Automatic mixed precision (fp16)

➢ Gradient accumulation

➢ Ensemble

Linear Learning rate decay

Method 1: Adjust learning rate manually

● Decrement optimizer.param_groups[0][“lr”] by learning_rate / total training step per step

●

●

1st window doc_stride is set to “max_paragraph_len” in sample code (i.e. no overlap) What if answer is near the boundary of windows or across windows? Hint: Overlapping windows

Preprocessing

Hint: How to prevent model from learning something it should not learn during training? (i.e. answers are not always near the middle of window)

Other pretrained models

You can choose any model you like! [Link to pretrained models in huggingface]

Note 1: You are NOT allowed to use pretrained models outside huggingface!

(Violation = cheating = final grade x 0.9)

Note 2: Some models have describing details of the model

Postprocessing

Hint: Open your prediction file to see what is wrong

(e.g. what if predicted end_index &lt; predicted start_index?)

Training Tip: Automatic mixed precision

● PyTorch trains with 32-bit floating point (FP32) arithmetic by default Estimated training time

T4 T4 (FP16)

Simple 20m 8m

Medium 20m 8m

Strong 1h 25m

Boss 6h 2.5h

● Automatic Mixed Precision (AMP) enables automatic conversion of certain GPU operations from FP32 precision to half-precision (FP16)

● Offer about 1.5-3.0x speed up while maintaining accuracy

(e.g. T4, V100)

Intro to native pytorch automatic mixed precision

Training Tip: Gradient accumulation

Use it when gpu memory is not enough but you want to use larger batch size

● Split global batch into smaller mini-batches

● For each mini-batch: Accumulate gradient without updating model parameters ● Update model parameters

Reference: Gradient Accumulation in PyTorch

4. Grading

Report +4pts

Code Submission +2pts

Simple Baseline (Public) +0.5pt

Simple Baseline (Private) +0.5pt

Medium Baseline (Public) +0.5pt

Medium Baseline (Private) +0.5pt

Strong Baseline (Public) +0.5pt

Strong Baseline (Private) +0.5pt

Boss Baseline (Public) +0.5pt

Boss Baseline (Private) +0.5pt

Grading

Kaggle Link:

Kaggle (4pts)

Displayed name:

Testing Set:

Evaluation Metric: www.kaggle.com/c/ml2022-spring-hw7

5/6 (Fri.) 23:59

&lt;student ID&gt;_&lt;anything&gt;

4957 Questions (~50% public set, ~50% private set)

Accuracy (Exact Match) Submission Format: (csv)

Baselines Public Score

Simple (0.5pt + 0.5pt) 0.45139

Medium (0.5pt + 0.5pt) 0.65792

Strong (0.5pt + 0.5pt) 0.78136

Boss (0.5pt + 0.5pt) 0.84388

Code Submission (2pts)

● NTU COOL

○ Compress your code into &lt;student ID&gt;_hw7.zip (e.g. b10901118_hw7.zip) ○ We can only see your last submission.

○ Do not submit your model or dataset.

○ If your code is not reasonable, your semester grade x 0.9.

Grading — Bonus

● If your ranking in private set is top 3, you can choose to share a report to NTU COOL and get extra 0.5 pts.

● About the report

○ Your name and student_ID ○ Methods you used in code

○ Reference Report Template

○ in 200 words

○ Please upload to NTU COOL’s discussion of HW7

5. Report

Report Questions (4 pts)

1. (2%) After your model predicts the probability of answer span start/end position, what rules did you apply to determine the final start/end position?

(the rules you applied must be different from the sample code)

2. (2%) Try another type of pretrained model which can be found in huggingface’s Model Hub (e.g. BERT -&gt; BERT-wwm-ext, or BERT -&gt; RoBERTa ), and describe

● the pretrained model you used

● performance of the pretrained model you used

● the difference between BERT and the pretrained model you used (architecture, pretraining loss, etc.)

6. Regulations

Regulations

● You should NOT plagiarize, if you use any other resource, you should cite it in the reference. (＊)

● You should NOT modify your prediction files manually.

● Do NOT share codes or prediction files with any living creatures.

● Do NOT use any approaches to submit your results more than 5 times a day.

● Do NOT search or use additional data.

● Do NOT use any pre-trained models outside huggingface.

● Your assignment will not be graded and your final grade x 0.9 if you violate any of the above rules.

● Prof. Lee &amp; TAs preserve the rights to change the rules &amp; grades.

(＊) Academic Ethics Guidelines for Researchers by the

Ministry of Science and Technology

Links

Kaggle : https://www.kaggle.com/c/ml2022spring-hw7

Colab :

https://colab.research.google.com/drive/1QloQ42zYwX_ETAs2GIkeh8q Fj0UjHXfH?usp=sharing

If any questions, you can ask us via…

● NTU COOL (recommended)

○ https://cool.ntu.edu.tw/courses/11666

● Email

○ The title should begin with “[hw7]”
