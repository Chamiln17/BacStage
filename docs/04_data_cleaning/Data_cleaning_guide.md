# Raw data before feature engineering
## 1. metadata (videos_bac_only.csv)

📊 Raw Videos (videos_bac_only.csv) Summary
============================================================
Shape: 10,030 rows × 21 columns
Memory: 16.60 MB
Duplicates: 0 rows

Data Types:
  - object: 12 columns
  - int64: 5 columns
  - datetime64[ns, UTC]: 1 columns
  - datetime64[ns]: 1 columns
  - bool: 1 columns
  - float64: 1 columns

⚠️ Missing Values (2 columns):
  - tags: 5,904.0 (58.9%)
  - description: 3,745.0 (37.3%)
📊 Missing Values Summary:
Total columns: 21
Columns with missing values: 2

Columns with missing values (sorted by count):
Column	Missing Count	Missing %	Non-Null Count
12	tags	5904	58.8634	4126
2	description	3745	37.3380	6285

### Descriptive statistics for key numerical columns
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>skew</th>
      <th>kurtosis</th>
      <th>null_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>view_count</th>
      <td>10030.0000</td>
      <td>93456.4900</td>
      <td>178949.3700</td>
      <td>111.0000</td>
      <td>12146.7500</td>
      <td>34807.5000</td>
      <td>97468.0000</td>
      <td>4063201.0000</td>
      <td>6.2700</td>
      <td>70.3000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>like_count</th>
      <td>10030.0000</td>
      <td>3731.3400</td>
      <td>7339.8500</td>
      <td>0.0000</td>
      <td>402.0000</td>
      <td>1327.0000</td>
      <td>3949.7500</td>
      <td>154733.0000</td>
      <td>6.1100</td>
      <td>63.2500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>comment_count</th>
      <td>10030.0000</td>
      <td>367.0300</td>
      <td>733.5400</td>
      <td>0.0000</td>
      <td>55.0000</td>
      <td>147.0000</td>
      <td>382.0000</td>
      <td>21201.0000</td>
      <td>7.7100</td>
      <td>113.6400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>duration_sec</th>
      <td>10030.0000</td>
      <td>1961.5300</td>
      <td>2788.2200</td>
      <td>120.0000</td>
      <td>642.0000</td>
      <td>1355.0000</td>
      <td>2427.0000</td>
      <td>72711.0000</td>
      <td>8.9200</td>
      <td>132.0700</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
🔍 Observations:
  ⚠️ view_count: Highly skewed (6.27) - consider log transformation
  ⚠️ like_count: Highly skewed (6.11) - consider log transformation
  ⚠️ comment_count: Highly skewed (7.71) - consider log transformation
  ⚠️ duration_sec: Highly skewed (8.92) - consider log transformation

 🔍 Outlier Detection (IQR Method):
------------------------------------------------------------
view_count:
  Bounds: [-115,835, 225,450]
  Outliers: 1,037 (10.3%)
like_count:
  Bounds: [-4,920, 9,271]
  Outliers: 997 (9.9%)
comment_count:
  Bounds: [-436, 872]
  Outliers: 981 (9.8%)
duration_sec:
  Bounds: [-2,036, 5,104]
  Outliers: 488 (4.9%) 

  <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Q1</th>
      <th>Q3</th>
      <th>IQR</th>
      <th>Lower Bound</th>
      <th>Upper Bound</th>
      <th>Outliers</th>
      <th>Outlier %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>view_count</td>
      <td>12146.7500</td>
      <td>97468.0000</td>
      <td>85321.2500</td>
      <td>-115835.1250</td>
      <td>225449.8750</td>
      <td>1037</td>
      <td>10.3390</td>
    </tr>
    <tr>
      <th>1</th>
      <td>like_count</td>
      <td>402.0000</td>
      <td>3949.7500</td>
      <td>3547.7500</td>
      <td>-4919.6250</td>
      <td>9271.3750</td>
      <td>997</td>
      <td>9.9402</td>
    </tr>
    <tr>
      <th>2</th>
      <td>comment_count</td>
      <td>55.0000</td>
      <td>382.0000</td>
      <td>327.0000</td>
      <td>-435.5000</td>
      <td>872.5000</td>
      <td>981</td>
      <td>9.7807</td>
    </tr>
    <tr>
      <th>3</th>
      <td>duration_sec</td>
      <td>642.0000</td>
      <td>2427.0000</td>
      <td>1785.0000</td>
      <td>-2035.5000</td>
      <td>5104.5000</td>
      <td>488</td>
      <td>4.8654</td>
    </tr>
  </tbody>
</table>
</div>

📊 Subject Distribution:
----------------------------------------
                     Count  Percent
subject                            
Maths                 4057  40.4500
History & Geography   1962  19.5600
Natural Sciences      1743  17.3800
Physics               1027  10.2400
Islamic Sciences       365   3.6400
Arabic                 357   3.5600
English                241   2.4000
French                 148   1.4800
Philosophy             130   1.3000

📊 Filter Category Distribution:
----------------------------------------
                   Count  Percent
filter_category                  
bac_3as             8755  87.2900
bac_3as_ambiguous   1275  12.7100

📊 Filter Confidence Distribution:
----------------------------------------
                   Count  Percent
filter_confidence                
0.9500              8755  87.2900
0.6500              1275  12.7100


### 📝 Text Feature Statistics:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title_length</th>
      <th>title_word_count</th>
      <th>description_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10030.0000</td>
      <td>10030.0000</td>
      <td>10030.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>60.8792</td>
      <td>10.3028</td>
      <td>185.3554</td>
    </tr>
    <tr>
      <th>std</th>
      <td>20.3863</td>
      <td>3.5030</td>
      <td>341.2764</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>45.0000</td>
      <td>8.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>60.0000</td>
      <td>10.0000</td>
      <td>85.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>76.0000</td>
      <td>13.0000</td>
      <td>255.0000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.0000</td>
      <td>24.0000</td>
      <td>4998.0000</td>
    </tr>
  </tbody>
</table>
</div>

![alt text](image.png)


### Key summary:
we should use log scaling for **view_count, like_count, comment_count and duration_sec** to handle the skewness.  
we should impute missing values for **description** with empty string.  
we should impute missing values for **tags** with empty string. 

## channels_statistics.csv
### channel coverage after filtering
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Channel</th>
      <th>Total Videos</th>
      <th>In Dataset</th>
      <th>Coverage %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>الاستاذ نورالدين</td>
      <td>5751</td>
      <td>3040</td>
      <td>52.8600</td>
    </tr>
    <tr>
      <th>35</th>
      <td>الأستاذ عبدالنور خليفي Abdennour Khalifi</td>
      <td>2442</td>
      <td>1289</td>
      <td>52.7800</td>
    </tr>
    <tr>
      <th>20</th>
      <td>- قناة الأستاذ سفيان - ط</td>
      <td>651</td>
      <td>623</td>
      <td>95.7000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>الاستاذة خيرة فليتي في علوم الطبيعة والحياة .</td>
      <td>610</td>
      <td>593</td>
      <td>97.2100</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Prof-Ahmed Trir</td>
      <td>1112</td>
      <td>592</td>
      <td>53.2400</td>
    </tr>
    <tr>
      <th>0</th>
      <td>الأستاذة كتفي شريف زينة لمادة علوم الطبيعةوالحياة</td>
      <td>787</td>
      <td>409</td>
      <td>51.9700</td>
    </tr>
    <tr>
      <th>34</th>
      <td>الأستاذ بورنان</td>
      <td>571</td>
      <td>397</td>
      <td>69.5300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mostafa bdd -العلوم الطبيعية</td>
      <td>440</td>
      <td>312</td>
      <td>70.9100</td>
    </tr>
    <tr>
      <th>30</th>
      <td>الأستاذة بوسعادي BOUSSAADi</td>
      <td>749</td>
      <td>225</td>
      <td>30.0400</td>
    </tr>
    <tr>
      <th>17</th>
      <td>الأستاذ عبد الباسط</td>
      <td>398</td>
      <td>214</td>
      <td>53.7700</td>
    </tr>
    <tr>
      <th>33</th>
      <td>الأستاذ عبدوش</td>
      <td>250</td>
      <td>213</td>
      <td>85.2000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Zeddoun Med El Amine</td>
      <td>190</td>
      <td>183</td>
      <td>96.3200</td>
    </tr>
    <tr>
      <th>13</th>
      <td>الاستاذ حيقون أسامة</td>
      <td>329</td>
      <td>171</td>
      <td>51.9800</td>
    </tr>
    <tr>
      <th>3</th>
      <td>الأستاذ بن عثمان علوم الطبيعة و الحياة</td>
      <td>240</td>
      <td>154</td>
      <td>64.1700</td>
    </tr>
    <tr>
      <th>19</th>
      <td>الاستاذ مرنيز وليد للرياضيات</td>
      <td>380</td>
      <td>146</td>
      <td>38.4200</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Nasri English</td>
      <td>419</td>
      <td>141</td>
      <td>33.6500</td>
    </tr>
    <tr>
      <th>14</th>
      <td>الأستاذ بوبكر مبروك اللغة العربية</td>
      <td>689</td>
      <td>140</td>
      <td>20.3200</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Dr_chms</td>
      <td>202</td>
      <td>140</td>
      <td>69.3100</td>
    </tr>
    <tr>
      <th>23</th>
      <td>sid ahmed chaallel</td>
      <td>547</td>
      <td>128</td>
      <td>23.4000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>الأستاذ عبد اللطيف Abdellatif</td>
      <td>527</td>
      <td>124</td>
      <td>23.5300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>الأستاذ شاوش</td>
      <td>363</td>
      <td>118</td>
      <td>32.5100</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Mr.Mansouri</td>
      <td>196</td>
      <td>114</td>
      <td>58.1600</td>
    </tr>
    <tr>
      <th>11</th>
      <td>amine english</td>
      <td>432</td>
      <td>88</td>
      <td>20.3700</td>
    </tr>
    <tr>
      <th>1</th>
      <td>الاستاذ باشا للعلوم</td>
      <td>397</td>
      <td>84</td>
      <td>21.1600</td>
    </tr>
    <tr>
      <th>5</th>
      <td>أيوب للعلوم</td>
      <td>80</td>
      <td>73</td>
      <td>91.2500</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Hind chaouaou</td>
      <td>99</td>
      <td>63</td>
      <td>63.6400</td>
    </tr>
    <tr>
      <th>29</th>
      <td>أستاذ الفلسفة عادل مقرود</td>
      <td>129</td>
      <td>57</td>
      <td>44.1900</td>
    </tr>
    <tr>
      <th>15</th>
      <td>الأستاذ شريفي عربية</td>
      <td>255</td>
      <td>46</td>
      <td>18.0400</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Prof rania</td>
      <td>235</td>
      <td>34</td>
      <td>14.4700</td>
    </tr>
    <tr>
      <th>25</th>
      <td>أستاذ الفلسفة. خليل سعيداني</td>
      <td>25</td>
      <td>24</td>
      <td>96.0000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>الفلسفة مع هواري</td>
      <td>28</td>
      <td>19</td>
      <td>67.8600</td>
    </tr>
    <tr>
      <th>28</th>
      <td>أستاذ الفلسفة أكرم غسان</td>
      <td>23</td>
      <td>18</td>
      <td>78.2600</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Reda.français</td>
      <td>107</td>
      <td>17</td>
      <td>15.8900</td>
    </tr>
    <tr>
      <th>8</th>
      <td>sally.français</td>
      <td>19</td>
      <td>17</td>
      <td>89.4700</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Mr. DJIAR Anglais</td>
      <td>56</td>
      <td>12</td>
      <td>21.4300</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Prof Fares philo</td>
      <td>21</td>
      <td>12</td>
      <td>57.1400</td>
    </tr>
  </tbody>
</table>
</div>

---

## Part 4: Transcript Analysis

Analyzing the transcript data to understand coverage, quality, and text characteristics.

📊 TRANSCRIPT AVAILABILITY ANALYSIS
===============================================================================     

✅ Successful transcripts: 4,622 (46.3%)
❌ Failed transcripts: 5,369 (53.7%)
📊 Total records: 9,991

![alt text](image-1.png)


❌ FAILURE REASON ANALYSIS (REFINED)
================================================================================

📋 Detailed Failure Breakdown:
------------------------------------------------------------
  Other/Unknown Error                      | 3,671 ( 68.4%)
  Bot Detection (Sign in required)         | 1,684 ( 31.4%)
  No Transcript Available                  |    11 (  0.2%)
  Private Video                            |     3 (  0.1%)
💾 Saved: transcript_failure_analysis.png


🌍 TRANSCRIPT LANGUAGE DISTRIBUTION
================================================================================

📋 Language Distribution (Available Transcripts):
------------------------------------------------------------
  ar                   | 4,581 (99.1%)
  live_chat            | 39 (0.8%)
  fr                   | 1 (0.0%)
  en                   | 1 (0.0%)
💾 Saved: transcript_languages.png


================================================================================
📏 TRANSCRIPT TEXT CHARACTERISTICS
================================================================================

📊 Character Count Statistics:
count      4622.0000
mean      29416.8300
std      110550.7200
min          50.0000
25%        6371.7500
50%       12754.5000
75%       23778.5000
max     1509363.0000
Name: char_count, dtype: float64

📊 Word Count Statistics:
count     4622.0000
mean      3842.8400
std       6438.9200
min          7.0000
25%       1173.5000
50%       2444.5000
75%       4585.0000
max     130579.0000
Name: word_count, dtype: float64
💾 Saved: transcript_text_lengths.png

📊 SEGMENT COUNT ANALYSIS
================================================================================

📊 Segment Count Statistics:
count   4622.0000
mean      32.7500
std      310.9000
min        1.0000
25%        1.0000
50%        1.0000
75%        2.0000
max     4219.0000
Name: segment_count, dtype: float64


📊 TRANSCRIPT COVERAGE BY SUBJECT
================================================================================

📋 Coverage by Subject:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>total_videos</th>
      <th>with_transcript</th>
      <th>coverage_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Maths</td>
      <td>4057</td>
      <td>1955</td>
      <td>48.1883</td>
    </tr>
    <tr>
      <th>3</th>
      <td>History &amp; Geography</td>
      <td>1962</td>
      <td>591</td>
      <td>30.1223</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Natural Sciences</td>
      <td>1743</td>
      <td>1132</td>
      <td>64.9455</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Physics</td>
      <td>1027</td>
      <td>439</td>
      <td>42.7459</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Islamic Sciences</td>
      <td>365</td>
      <td>94</td>
      <td>25.7534</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Arabic</td>
      <td>357</td>
      <td>110</td>
      <td>30.8123</td>
    </tr>
    <tr>
      <th>1</th>
      <td>English</td>
      <td>241</td>
      <td>108</td>
      <td>44.8133</td>
    </tr>
    <tr>
      <th>2</th>
      <td>French</td>
      <td>148</td>
      <td>69</td>
      <td>46.6216</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Philosophy</td>
      <td>130</td>
      <td>124</td>
      <td>95.3846</td>
    </tr>
  </tbody>
</table>
</div>

### summary
maybe we should also use log scale for word count and character count since it's right skewed too


## engineering analysis


🎯 TARGET VARIABLE ANALYSIS: engagement_score
================================================================================

📊 Basic Statistics:
------------------------------------------------------------
  Count: 10,030
  Mean: 2.4194
  Median: 2.4525
  Std: 0.7418
  Min: 0.0000
  Max: 5.1399
  Skewness: -0.1249
  Kurtosis: -0.1228

📊 Outlier Analysis (IQR Method):
------------------------------------------------------------
  Q1: 1.9361
  Q3: 2.9157
  IQR: 0.9796
  Lower bound: 0.4667
  Upper bound: 4.3852
  Outliers: 48 (0.5%)

🎯 Top 20 Correlations with engagement_score:
------------------------------------------------------------
  channel_avg_views                        | +0.3982
  channel_video_count                      | -0.3005
  channel_age_days                         | -0.2773
  channel_video_avg_duration               | -0.1838
  tag_count                                | -0.1786
  comment_ratio                            | +0.1733
  technical_term_density                   | -0.1336
  question_density                         | -0.1267
  title_word_count                         | +0.1235
  is_exam_focused                          | +0.0994
  contrast_density                         | -0.0957
  title_length                             | +0.0845
  duration_sec                             | +0.0764
  question_count                           | -0.0758
  avg_words_per_sentence                   | -0.0668
  example_count                            | +0.0561
  explanation_density                      | +0.0479
  explanation_count                        | +0.0470
  technical_term_count                     | -0.0430
  transcript_sentence_count                | -0.0321


⚠️ MULTICOLLINEARITY CHECK
================================================================================

📋 Highly Correlated Feature Pairs (|corr| > 0.8):
   Found 10 pairs

   <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature_1</th>
      <th>Feature_2</th>
      <th>Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>transcript_word_count</td>
      <td>transcript_char_count</td>
      <td>0.9982</td>
    </tr>
    <tr>
      <th>4</th>
      <td>channel_video_count</td>
      <td>channel_age_days</td>
      <td>0.9464</td>
    </tr>
    <tr>
      <th>2</th>
      <td>title_length</td>
      <td>title_word_count</td>
      <td>0.8951</td>
    </tr>
    <tr>
      <th>3</th>
      <td>view_count</td>
      <td>like_count</td>
      <td>0.8695</td>
    </tr>
    <tr>
      <th>9</th>
      <td>transcript_char_count</td>
      <td>contrast_count</td>
      <td>0.8664</td>
    </tr>
    <tr>
      <th>7</th>
      <td>transcript_word_count</td>
      <td>contrast_count</td>
      <td>0.8600</td>
    </tr>
    <tr>
      <th>8</th>
      <td>transcript_char_count</td>
      <td>unique_word_count</td>
      <td>0.8570</td>
    </tr>
    <tr>
      <th>6</th>
      <td>transcript_word_count</td>
      <td>unique_word_count</td>
      <td>0.8407</td>
    </tr>
    <tr>
      <th>1</th>
      <td>duration_sec</td>
      <td>transcript_char_count</td>
      <td>0.8351</td>
    </tr>
    <tr>
      <th>0</th>
      <td>duration_sec</td>
      <td>transcript_word_count</td>
      <td>0.8348</td>
    </tr>
  </tbody>
</table>
</div>

💡 Recommendation: Consider removing one feature from each highly correlated pair to reduce multicollinearity

📊 PERFORMANCE: WITH vs WITHOUT TRANSCRIPTS
================================================================================

📋 Engagement Metrics Comparison:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">view_count</th>
      <th colspan="2" halign="left">like_count</th>
      <th colspan="2" halign="left">comment_count</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>transcript_available</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>91591.0700</td>
      <td>33608.5000</td>
      <td>180433.5600</td>
      <td>3848.0700</td>
      <td>1521.5000</td>
      <td>327.0900</td>
      <td>145.5000</td>
    </tr>
    <tr>
      <th>True</th>
      <td>95639.1400</td>
      <td>36576.0000</td>
      <td>177191.5600</td>
      <td>3594.7800</td>
      <td>1109.5000</td>
      <td>413.7600</td>
      <td>148.5000</td>
    </tr>
  </tbody>
</table>
</div>

## general recommendations 
Issue	Severity	Percentage	Affected_Rows	Recommendation
- 0	Missing description	Medium	37.3%	3745	Fill with empty string
- 1	Missing transcripts	High	53.9%	5408	Add has_transcript flag, fill derived features with 0
- 2	Missing readability features	High	54.3%	5447	Fill with 0 (as these likely map to no transcript)
3	
- Highly skewed features (>2 skew)	Medium	N/A	27 features	Apply Log1p transformation