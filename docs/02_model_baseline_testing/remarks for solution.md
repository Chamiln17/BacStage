# recomendations from mentors
- each feature must be mesearable 
- it needs a script to extract the features from the video directly 
- the content must be features not the dates
- If the video is published : they give the url of video 
  - else they gave access to draft video  could be perspective
- Video characteristics should be extracted directly (by upload)
- some models characteriscts automiatically
- engagment rate what does mean (in this case type of videos, learning because of this catergory is valuable because you need effort and type of people is different than other type of videos.)
- sometimes doesn't mean a channel 


# idea
we should focus on extracting content related features from the video (either by uploading the video inside our system in production or by scraping for training because post posting , the content crearor)

i got some remarks from the mentors about the idea. i want you to undersand it and improve it since the solution should focus more on pre engagemnt rate prediction. do comprehensive search

### MVP Metric: Enhanced Weighted Engagement Score

Based on the provided document, the recommended Minimum Viable Product (MVP) metric for labeling educational video engagement is the **Enhanced Weighted Engagement Score**.

#### The Formula

$$\text{Engagement Score} = \left( 0.40 \times \frac{\text{Likes}}{\text{Views}} + 0.50 \times \frac{\text{Comments}}{\text{Views}} + 0.10 \times \text{Age Factor} \right) \times 100$$

#### Component Definitions

* 
**Comments/Views (50% Weight):** This is weighted heavily because comments are the strongest signal for educational engagement, indicating active learning and cognitive processing.


* 
**Likes/Views (40% Weight):** Serves as a signal of general approval.


* 
**Age Factor (10% Weight):** Normalizes the score to prevent recency bias or the domination of old viral videos. It is calculated as:


$$\text{Age Factor} = \min\left(1.0, \frac{\text{Days Since Publish}}{365}\right)$$





#### Output

This formula generates a continuous score on a **0–100 scale**, making it interpretable and suitable for regression models.