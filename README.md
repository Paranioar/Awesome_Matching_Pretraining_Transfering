Cross-modal_Retrieval_Tutorial
==============================
The Tutorial of Image-Text Matching for Preliminary Insight. 
****
## ``Catalogue ``
* [Peformance comparison](#peformance-comparison)
    * [Flickr8K](#performance-of-flickr8k)
    * [Flickr30K](#performance-of-flickr30k)
    * [MSCOCO1K](#performance-of-mscoco1k)
    * [MSCOCO5K](#performance-of-mscoco5k)
    * [CUHK-PEDES](#performance-of-cukh-pedes)
    * [CUB-Flowers](#performance-of-cub-flowers)

* [Methods summary](#method-summary)
    * [Generic-feature extraction](#generic-feature-extraction)
    * [Cross-modal interaction](#cross-modal-interaction)
    * [Similarity measurement](#similarity-measurement)
    * [Loss function](#loss-function)
    * [Un-supervised or Semi-supervised](#un-supervised-or-semi-supervised)
    * [Zero-shot or Fewer-shot](#zero-shot-or-fewer-shot)
    * [Adversarial learning](#adversarial-learning)
    * [Identification learning](#identification-learning)
    * [Related works](#related-works)
    * [Posted in](#posted-in)
****

## ``Peformance comparison``

### *Performance of Flickr8K*
**(*\** indicates Ensemble models, *^* indicates questionable authen)**
<table>
   <tr> <td rowspan="3">Method_name</td> <td rowspan="3", align="center">Concise_note</td> 
        <td colspan="6", align="center">Flickr8K</td> </tr>
   <tr> <td colspan="3", align="center">Sentence retrieval</td> <td colspan="3", align="center">Image retrieval</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td> <td>R@1</td><td>R@5</td><td>R@10</td> </tr>
   <tr> <td>DeViSE</td><td>RCNN</td> <td>4.8</td><td>16.5</td><td>27.3</td> <td>5.9</td><td>20.1</td><td>29.6</td> </tr>
   <tr> <td>SDT-RNN</td><td>AlexNet</td> <td>4.5</td><td>18.0</td><td>28.6</td> <td>6.1</td><td>18.5</td><td>29.0</td> </tr> 
   <tr> <td>SDT-RNN</td><td>RCNN</td> <td>6.0</td><td>22.7</td><td>34.0</td> <td>6.6</td><td>21.6</td><td>31.7</td> </tr>
   <tr> <td>DeFrag</td><td>AlexNet</td> <td>5.9</td><td>19.2</td><td>27.3</td> <td>5.2</td><td>17.6</td><td>26.5</td> </tr>
   <tr> <td>DeFrag</td><td>RCNN</td> <td>12.6</td><td>32.9</td><td>44.0</td> <td>9.7</td><td>29.6</td><td>42.5</td> </tr>
   <tr> <td>m-RNN</td><td>AlexNet</td> <td>14.5</td><td>37.2</td><td>48.5</td> <td>11.5</td><td>31.0</td><td>42.4</td> </tr>
   <tr> <td>DVSA</td><td>DepTree</td> <td>14.8</td><td>37.9</td><td>50.0</td> <td>11.6</td><td>31.4</td><td>43.8</td> </tr>
   <tr> <td>DVSA</td><td>RCNN</td> <td>16.5</td><td>40.6</td><td>54.2</td> <td>11.8</td><td>32.1</td><td>44.7</td> </tr>
   <tr> <td>UVSE</td><td>AlexNet</td> <td>13.5</td><td>36.2</td><td>45.7</td> <td>10.4</td><td>31.0</td><td>43.7</td> </tr>
   <tr> <td>UVSE</td><td>VggNet</td> <td>18.0</td><td>40.9</td><td>55.0</td> <td>12.5</td><td>37.0</td><td>51.5</td> </tr>
   <tr> <td>NIC</td><td>GoogleNet</td> <td>20</td><td>--</td><td>61</td> <td>19</td><td>--</td><td>64</td> </tr>
   <tr> <td>m-CNN*</td><td>OverFeat</td> <td>14.9</td><td>35.9</td><td>49.0</td> <td>11.8</td><td>34.5</td><td>48.0</td> </tr>
   <tr> <td>m-CNN*</td><td>VggNet</td> <td>24.8</td><td>53.7</td><td>67.1</td> <td>20.3</td><td>47.6</td><td>61.7</td> </tr>
   <tr> <td>HM-LSTM</td><td>RCNN</td> <td>27.7</td><td>--</td><td>68.6</td> <td>24.4</td><td>--</td><td>68.1</td> </tr>
   <tr> <td>FV</td><td>GMM+HGLMM</td> <td>31.0</td><td>59.3</td><td>73.7</td> <td>21.2</td><td>50.0</td><td>64.8</td> </tr>
   <tr> <td>SCAN*</td><td>BUTD</td> <td>52.2</td><td>81.0</td><td>89.2</td> <td>38.3</td><td>67.8</td><td>78.9</td> </tr>
   <tr> <td>IMRAM_t2i</td><td>BUTD</td> <td>48.5</td><td>78.1</td><td>85.3</td> <td>32.0</td><td>61.4</td><td>73.9</td> </tr>
   <tr> <td>IMRAM_i2t</td><td>BUTD</td> <td>52.1</td><td>81.5</td><td>90.1</td> <td>40.2</td><td>69.0</td><td>79.2</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD</td> <td>54.7</td><td>84.2</td><td>91.0</td> <td>41.0</td><td>69.2</td><td>79.9</td> </tr>
</table> 

### *Performance of Flickr30K*
<table>
   <tr> <td rowspan="3">Method_name</td> <td rowspan="3", align="center">Concise_note</td> 
        <td colspan="6", align="center">Flickr30K</td> </tr>
   <tr> <td colspan="3", align="center">Sentence retrieval</td> <td colspan="3", align="center">Image retrieval</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td> <td>R@1</td><td>R@5</td><td>R@10</td> </tr>
   <tr> <td>DeViSE</td><td>RCNN</td> <td>4.5</td><td>18.1</td><td>29.2</td> <td>6.7</td><td>21.9</td><td>32.7</td> </tr>
   <tr> <td>SDT-RNN</td><td>RCNN</td> <td>9.6</td><td>29.8</td><td>41.1</td> <td>8.9</td><td>29.8</td><td>41.1</td> </tr>
   <tr> <td>DeFrag</td><td>RCNN</td> <td>14.2</td><td>37.7</td><td>51.3</td> <td>10.2</td><td>30.8</td><td>44.2</td> </tr>
   <tr> <td>DeFrag</td><td>ftCNN</td> <td>16.4</td><td>40.2</td><td>54.7</td> <td>10.3</td><td>31.4</td><td>44.5</td> </tr>
   <tr> <td>NIC</td><td>GoogleNet</td> <td>17</td><td>--</td><td>56</td> <td>17</td><td>--</td><td>57</td> </tr>
   <tr> <td>DVSA</td><td>DepTree</td> <td>20.0</td><td>46.6</td><td>59.4</td> <td>15.0</td><td>36.5</td><td>48.2</td> </tr>
   <tr> <td>DVSA</td><td>RCNN</td> <td>22.2</td><td>48.2</td><td>61.4</td> <td>15.2</td><td>37.7</td><td>50.5</td> </tr>
   <tr> <td>UVSE</td><td>AlexNet</td> <td>14.8</td><td>39.2</td><td>50.9</td> <td>11.8</td><td>34.0</td><td>46.3</td> </tr>
   <tr> <td>UVSE</td><td>VggNet</td> <td>23.0</td><td>50.7</td><td>62.9</td> <td>16.8</td><td>42.0</td><td>56.5</td> </tr>
   <tr> <td>m-RNN</td><td>AlexNet</td> <td>18.4</td><td>40.2</td><td>50.9</td> <td>12.6</td><td>31.2</td><td>41.5</td> </tr>
   <tr> <td>m-RNN</td><td>VggNet</td> <td>35.4</td><td>63.8</td><td>73.7</td> <td>22.8</td><td>50.7</td><td>63.1</td> </tr>
   <tr> <td>FV</td><td>GMM+HGLMM</td> <td>35.0</td><td>62.0</td><td>73.8</td> <td>25.0</td><td>52.7</td><td>66.0</td> </tr>
   <tr> <td>m-CNN*</td><td>OverFeat</td> <td>20.1</td><td>44.2</td><td>56.3</td> <td>15.9</td><td>40.3</td><td>51.9</td> </tr>
   <tr> <td>m-CNN*</td><td>VggNet</td> <td>33.6</td><td>64.1</td><td>74.9</td> <td>26.2</td><td>56.3</td><td>69.6</td> </tr>
   <tr> <td>HM-LSTM</td><td>RCNN</td> <td>38.1</td><td>--</td><td>76.5</td> <td>27.7</td><td>--</td><td>68.8</td> </tr>
   <tr> <td>SPE</td><td>VggNet</td> <td>40.3</td><td>68.9</td><td>79.9</td> <td>29.7</td><td>60.1</td><td>72.1</td> </tr>
   <tr> <td>sm-LSTM</td><td>VggNet</td> <td>42.4</td><td>67.5</td><td>79.9</td> <td>28.2</td><td>57.0</td><td>68.4</td> </tr>
   <tr> <td>sm-LSTM*</td><td>VggNet</td> <td>42.5</td><td>71.9</td><td>81.5</td> <td>30.2</td><td>60.4 </td><td>72.3</td> </tr>
   <tr> <td>CSE</td><td>ResNet</td> <td>44.6</td><td>74.3</td><td>83.8</td> <td>36.9</td><td>69.1</td><td>79.6</td> </tr>
   <tr> <td>2WayNet</td><td>VggNet</td> <td>49.8</td><td>67.5</td><td>--</td> <td>36.0</td><td>55.6</td><td>--</td> </tr>
   <tr> <td>RRF-Net</td><td>ResNet</td> <td>47.6</td><td>77.4</td><td>87.1</td> <td>35.4</td><td>68.3</td><td>79.9</td> </tr>
   <tr> <td>CMPL</td><td>MobileNet</td> <td>40.3</td><td>66.9</td><td>76.7</td> <td>30.4</td><td>58.2</td><td>68.5</td> </tr>
   <tr> <td>CMPL</td><td>ResNet</td> <td>49.6</td><td>76.8</td><td>86.1</td> <td>37.3</td><td>65.7</td><td>75.5</td> </tr>
   <tr> <td>VSE++</td><td>VggNet</td> <td>41.3</td><td>69.1</td><td>77.9</td> <td>31.4</td><td>60.0</td><td>71.2</td> </tr>
   <tr> <td>VSE++</td><td>ResNet</td> <td>52.9</td><td>80.5</td><td>87.2</td> <td>39.6</td><td>70.1</td><td>79.5</td> </tr>
   <tr> <td>TIMAM</td><td>ResNet, Bert</td> <td>53.1</td><td>78.8</td><td>87.6</td> <td>42.6</td><td>71.6</td><td>81.9</td> </tr>
   <tr> <td>DAN</td><td>VggNet</td> <td>41.4</td><td>73.5</td><td>82.5</td> <td>31.8</td><td>61.7</td><td>72.5</td> </tr>
   <tr> <td>DAN</td><td>ResNet</td> <td>55.0</td><td>81.8</td><td> 89.0</td> <td>39.4</td><td>69.2</td><td>79.1</td> </tr>
   <tr> <td>SCAN_t2i</td><td>BUTD, AVE</td> <td>61.8</td><td>87.5</td><td>93.7</td> <td>45.8</td><td>74.4</td><td>83.0</td> </tr>
   <tr> <td>SCAN_i2t</td><td>BUTD, AVE</td> <td>67.9</td><td>89.0</td><td>94.4</td> <td>43.9</td><td>74.2</td><td>82.8</td> </tr>
   <tr> <td>SCAN*</td><td>BUTD, AVE+LSE</td> <td>67.4</td><td>90.3</td><td>95.8</td> <td>48.6</td><td>77.7</td><td>85.2</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   

</table> 

### *Performance of MSCOCO1K*
<table>
   <tr> <td rowspan="3">Method_name</td> <td rowspan="3", align="center">Concise_note</td> 
        <td colspan="6", align="center">MSCOCO1K</td> </tr>
   <tr> <td colspan="3", align="center">Sentence retrieval</td> <td colspan="3", align="center">Image retrieval</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td> <td>R@1</td><td>R@5</td><td>R@10</td> </tr>
   <tr> <td>STV</td><td>combine-skip</td> <td>33.8</td><td>67.7</td><td>82.1</td> <td>25.9</td><td>60.0</td><td>74.6</td> </tr>
   <tr> <td>FV</td><td>GMM+HGLMM</td> <td>39.4</td><td>67.9</td><td>80.9</td> <td>25.1</td><td>59.8</td><td>76.6</td> </tr>
   <tr> <td>DVSA</td><td>RCNN</td> <td>38.4</td><td>69.9</td><td>80.5</td> <td>27.4</td><td>60.2</td><td>74.8</td> </tr>
   <tr> <td>m-RNN</td><td>VggNet</td> <td>41.0</td><td>73.0</td><td>83.5</td> <td>29.0</td><td>42.2</td><td>77.0</td> </tr>
   <tr> <td>m-CNN*</td><td>VggNet</td> <td>42.8</td><td>73.1</td><td>84.1</td> <td>32.6</td><td>68.6</td><td>82.8</td> </tr>
   <tr> <td>UVSE</td><td>VggNet</td> <td>43.4</td><td>75.7</td><td>85.8</td> <td>31.0</td><td>66.7</td><td>79.9</td> </tr>
   <tr> <td>HM-LSTM</td><td>RCNN</td> <td>43.9</td><td>--</td><td>87.8</td> <td>36.1</td><td>--</td><td>86.7</td> </tr>
   <tr> <td>Order-emb</td><td>VggNet</td> <td>46.7</td><td>--</td><td>88.9</td> <td>37.9</td><td>--</td><td>85.9</td> </tr>
   <tr> <td>SPE</td><td>VggNet</td> <td>50.1</td><td>79.7</td><td>89.2</td> <td>39.6</td><td>75.2</td><td>86.9</td> </tr> 
   <tr> <td>SEAM</td><td>VggNet</td> <td>50.7</td><td>81.4</td><td>90.9</td> <td>40.3</td><td>75.7</td><td>87.4</td> </tr>
   <tr> <td>sm-LSTM</td><td>VggNet</td> <td>52.4</td><td>81.7</td><td>90.8</td> <td>38.6</td><td>73.4</td><td>84.6</td> </tr>
   <tr> <td>sm-LSTM*</td><td>VggNet</td> <td>53.2</td><td>83.1</td><td>91.5</td> <td>40.7</td><td>75.8</td><td>87.4</td> </tr>  
   <tr> <td>2WayNet</td><td>VggNet</td> <td>55.8</td><td>75.2</td><td>--</td> <td>39.7</td><td>63.3</td><td>--</td> </tr>
   <tr> <td>CMPL</td><td>MobileNet</td> <td>52.9</td><td>83.8</td><td>92.1</td> <td>41.3</td><td>74.6</td><td>85.9</td> </tr>
   <tr> <td>CMPM</td><td>ResNet</td> <td>56.1</td><td>86.3</td><td>92.9</td> <td>44.6</td><td>78.8</td><td>89.0</td> </tr>
   <tr> <td>RRF-Net</td><td>ResNet</td> <td>56.4</td><td>85.3</td><td>91.5</td> <td>43.9</td><td>78.1</td><td>88.6</td> </tr>   
   <tr> <td>CSE</td><td>ResNet</td> <td>56.3</td><td>84.4</td><td>92.2</td> <td>45.7</td><td>81.2</td><td>90.6</td> </tr>    
   <tr> <td>CHAIN-VSE</td><td>VggNet</td> <td>51.6</td><td>82.0</td><td>91.3</td> <td>38.6</td><td>75.1</td><td>87.2</td> </tr>
   <tr> <td>CHAIN-VSE</td><td>ResNet</td> <td>59.4</td><td>88.0</td><td>94.2</td> <td>43.5</td><td>79.8</td><td>90.2</td> </tr>
   <tr> <td>VSE++</td><td>VggNet</td> <td>57.2</td><td>86.0</td><td>93.3</td> <td>45.9</td><td>79.4</td><td>89.1</td> </tr>
   <tr> <td>VSE++</td><td>ResNet</td> <td>64.6</td><td>90.0</td><td>95.7</td> <td>52.0</td><td>84.3</td><td>92.0</td> </tr>
   <tr> <td>SCAN_t2i</td><td>BUTD, AVE</td> <td>70.9</td><td>94.5</td><td>97.8</td> <td>56.4</td><td>87.0</td><td>93.9</td> </tr>
   <tr> <td>SCAN_i2t</td><td>BUTD, AVE</td> <td>69.2</td><td>93.2</td><td>97.5</td> <td>54.4</td><td>86.0</td><td>93.6</td> </tr>
   <tr> <td>SCAN*</td><td>BUTD, LSE+AVE</td> <td>72.7</td><td>94.8</td><td>98.4</td> <td>58.8</td><td>88.4 </td><td>94.8</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
</table> 

### *Performance of MSCOCO5K*
<table>
   <tr> <td rowspan="3">Method_name</td> <td rowspan="3", align="center">Concise_note</td> 
        <td colspan="6", align="center">MSCOCO5K</td> </tr>
   <tr> <td colspan="3", align="center">Sentence retrieval</td> <td colspan="3", align="center">Image retrieval</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td> <td>R@1</td><td>R@5</td><td>R@10</td> </tr>
   <tr> <td>DVSA</td><td>RCNN</td> <td>16.5</td><td>39.2</td><td>52.0</td> <td>10.7</td><td>29.6</td><td>42.2</td> </tr>
   <tr> <td>FV</td><td>GMM+HGLMM</td> <td>17.3</td><td>39.0</td><td>50.2</td> <td>10.8</td><td>28.3</td><td>40.1</td> </tr>
   <tr> <td>Order-emb</td><td>VggNet</td> <td>23.3</td><td>--</td><td>65.0</td> <td>18.0</td><td>--</td><td>57.6</td> </tr>
   <tr> <td>CSE</td><td>ResNet</td> <td>27.9</td><td>57.1</td><td>70.4</td> <td>22.2</td><td>50.2</td><td>64.4</td> </tr>
   <tr> <td>CMPL</td><td>MobileNet</td> <td>24.6</td><td>52.3</td><td>66.4</td> <td>19.1</td><td>44.6</td><td>58.4</td> </tr>
   <tr> <td>CMPM</td><td>ResNet</td> <td>31.1</td><td>60.7</td><td>73.9</td> <td>22.9</td><td>50.2</td><td>63.8</td> </tr>
   <tr> <td>VSE++</td><td>VggNet</td> <td>32.9</td><td>61.7</td><td>74.7</td> <td>24.1</td><td>52.8</td><td>66.2</td> </tr>
   <tr> <td>VSE++</td><td>ResNet</td> <td>41.3</td><td>71.1</td><td>81.2</td> <td>30.3</td><td>59.4</td><td>72.4</td> </tr>
   <tr> <td>SCAN_i2t</td><td>BUTD, LSE</td> <td>46.4</td><td>77.4</td><td>87.2</td> <td>34.4</td><td>63.7</td><td>75.7</td> </tr>
   <tr> <td>SCAN*</td><td>BUTD, AVE+LSE</td> <td>50.4</td><td>82.2</td><td>90.0</td> <td>38.6</td><td>69.3</td><td>80.4</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
   <tr> <td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> <td>222</td><td>222</td><td>222</td> </tr>
</table> 

### *performance of CUHK-PEDES*
<table>
   <tr> <td rowspan="2">Method_name</td> <td rowspan="2", align="center">Concise_note</td> 
        <td colspan="3", align="center">CUHK-PEDES</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td></tr>
   <tr> <td>LSTM-Q+I</td><td>VggNet</td> <td>17.19</td><td>--</td><td>57.82</td> </tr>
   <tr> <td>GNA-RNN</td><td>VggNet</td> <td>19.05</td><td>--</td><td>53.64</td> </tr>
   <tr> <td>IATV</td><td>VggNet</td> <td>25.94</td><td>--</td><td>60.48</td> </tr>
   <tr> <td>PWM-ATH</td><td>VggNet</td> <td>27.14</td><td>49.45</td><td>61.02</td> </tr>
   <tr> <td>GLA</td><td>ResNet</td> <td>43.58</td><td>66.93</td><td>76.26</td> </tr>
   <tr> <td>Dual-Path</td><td>VggNet</td> <td>32.15</td><td>54.42</td><td>64.30</td> </tr>
   <tr> <td>Dual-Path</td><td>ResNet</td> <td>44.40</td><td>66.26</td><td>75.07</td> </tr>
   <tr> <td>CMPM</td><td>MobileNet</td> <td>44.02</td><td>--</td><td>77.00</td> </tr>
   <tr> <td>CMPL</td><td>MobileNet</td> <td>49.37</td><td>--</td><td>79.27</td> </tr>
   <tr> <td>PMA</td><td>VggNet</td> <td>47.02</td><td>68.54</td><td>78.06</td> </tr>
   <tr> <td>PMA</td><td>ResNet</td> <td>53.81</td><td>73.54</td><td>81.23</td> </tr>
   <tr> <td>TIMAM</td><td>ResNet, Bert</td> <td>54.51</td><td>77.56</td><td>84.78</td> </tr>
</table> 

### *performance of CUB-Flowers*
<table>
   <tr> <td rowspan="3">Method_name</td> <td rowspan="3", align="center">Concise_note</td> 
        <td colspan="6", align="center">CUB-Flowers</td> </tr>
   <tr> <td align="center">Image-to-Text</td> <td align="center">Text-to-Image</td> 
        <td align="center">Image-to-Text</td> <td align="center">Text-to-Image</td> </tr>
   <tr> <td>R@1</td><td>AP@50</td> <td>R@1</td><td>AP@50</td> </tr>
   <tr> <td>Word2Vec</td><td></td> <td>38.6</td><td>33.5</td> <td>54.2</td><td>52.1</td> </tr>
   <tr> <td>FV</td><td>GMM+HGLMM</td> <td>36.5</td><td>35.6</td> <td>54.8</td><td>52.8</td> </tr>
   <tr> <td>Triplet</td><td>IATV</td> <td>52.5</td><td>52.4</td> <td>64.3</td><td>64.9</td> </tr>
   <tr> <td>Word-NN</td><td>CNN</td> <td>51.0</td><td>43.3</td> <td>60.7</td><td>56.3</td> </tr>
   <tr> <td>Word-NN</td><td>CNN-RNN</td> <td>56.8</td><td>48.7</td> <td>65.6</td><td>59.6</td> </tr>
   <tr> <td>IATV</td><td>VggNet</td> <td>61.5</td><td>57.6</td> <td>68.4</td><td>70.1</td> </tr>
   <tr> <td>CMPM</td><td>MobileNet</td> <td>62.1</td><td>64.6</td> <td>66.1</td><td>67.7</td> </tr>
   <tr> <td>CMPL</td><td>MobileNet</td> <td>64.3</td><td>67.9</td> <td>68.9</td><td>69.7</td> </tr>
   <tr> <td>TIMAM</td><td>ResNet, Bert</td> <td>67.7</td><td>70.3</td> <td>70.6</td><td>73.7</td> </tr>
</table> 

****

## ``Method summary`` 

### ``*Generic-feature extraction*``
**(*DeViSE_NIPS2013*) DeViSE: A Deep Visual-Semantic Embedding Model.** <br>
*Andrea Frome, Greg S. Corrado, Jonathon Shlens, Samy Bengio, Jeffrey Dean, Marc’Aurelio Ranzato, Tomas Mikolov.*<br>
[[paper]](https://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf)

**(*SDT-RNN_TACL2014*) Grounded Compositional Semantics for Finding and Describing Images with Sentences.**<br>
*Richard Socher, Andrej Karpathy, Quoc V. Le, Christopher D. Manning, Andrew Y. Ng.*<br>
[[paper]](https://www.aclweb.org/anthology/Q14-1017.pdf)

**(*DeFrag_NIPS2014*) Deep fragment embeddings for bidirectional image sentence mapping.**<br>
*Andrej Karpathy, Armand Joulin, Li Fei-Fei.*<br>
[[paper]](https://cs.stanford.edu/people/karpathy/nips2014.pdf)

**(*UVSE_NIPSws2014*) Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models.**<br>
*Ryan Kiros, Ruslan Salakhutdinov, Richard S. Zemel.*<br>
[[paper]](https://arxiv.org/pdf/1411.2539.pdf)
[[code]](https://github.com/ryankiros/visual-semantic-embedding)
[[demo]](http://www.cs.toronto.edu/~rkiros/lstm_scnlm.html)

**(*m-CNN_ICCV2015*) Multimodal Convolutional Neural Networks for Matching Image and Sentence.**<br>
*Lin Ma, Zhengdong Lu, Lifeng Shang, Hang Li.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7410658)

**(*STV_NIPS2015*) Skip-thought Vectors.**<br>
*Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, Sanja Fidler.*<br>
[[paper]](https://arxiv.org/pdf/1506.06726)

**(*DCCA_CVPR2015*) Deep Correlation for Matching Images and Text.**<br>
*Fei Yan, Krystian Mikolajczyk.*<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2015/papers/Yan_Deep_Correlation_for_2015_CVPR_paper.pdf)

**(*FV_CVPR2015*) Associating Neural Word Embeddings with Deep Image Representationsusing Fisher Vectors.**<br>
*Benjamin Klein, Guy Lev, Gil Sadeh, Lior Wolf.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7299073)

**(*DVSA_CVPR2015*) Deep Visual-Semantic Alignments for Generating Image Descriptions.**<br>
*Andrej Karpathy, Li Fei-Fei.*<br>
[[paper]](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)

**(*SPE_CVPR2016*) Learning Deep Structure-Preserving Image-Text Embeddings.**<br>
*Liwei Wang, Yin Li, Svetlana Lazebnik.*<br>
[[paper]](http://slazebni.cs.illinois.edu/publications/cvpr16_structure.pdf)

**(*2WayNet_CVPR2017*) Linking Image and Text with 2-Way Nets.**<br>
*Aviv Eisenschtat, Lior Wolf.*<br>
[[paper]](https://arxiv.org/pdf/1608.07973)

**(*HM-LSTM_ICCV2017*) Hierarchical Multimodal LSTM for Dense Visual-Semantic Embedding.**<br>
*Zhenxing Niu, Mo Zhou, Le Wang, Xinbo Gao, Gang Hua.*<br>
[[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Niu_Hierarchical_Multimodal_LSTM_ICCV_2017_paper.pdf)

**(*RRF-Net_ICCV2017*) Learning a Recurrent Residual Fusion Network for Multimodal Matching.**<br>
*Yu Liu, Yanming Guo, Erwin M. Bakker, Michael S. Lew.*<br>
[[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_a_Recurrent_ICCV_2017_paper.pdf)

**(*SEAM_WACV2018*) Fast Self-Attentive Multimodal Retrieval.**<br>
*Jônatas Wehrmann, Maurício Armani Lopes, Martin D More, Rodrigo C. Barros.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8354311&tag=1)
[[code]](https://github.com/jwehrmann/seam-retrieval)

**(*CSE_CVPR2018*) End-to-end Convolutional Semantic Embeddings.**<br>
*Quanzeng You, Zhengyou Zhang, Jiebo Luo.*<br>
[[paper]](https://ai.tencent.com/ailab/media/publications/cvpr/End-to-end_Convolutional_Semantic_Embeddings.pdf)

**(*CHAIN-VSE_CVPR2018*) Bidirectional Retrieval Made Simple.**<br>
*Jonatas Wehrmann, Rodrigo C. Barros.*<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wehrmann_Bidirectional_Retrieval_Made_CVPR_2018_paper.pdf)
[[code]](https://github.com/jwehrmann/chain-vse)

**(*SCO_CVPR2018*) Learning Semantic Concepts and Order for Image and Sentence Matching.**<br>
*Yan Huang, Qi Wu, Liang Wang.*<br>
[[paper]](https://arxiv.org/pdf/1712.02036)

### ``*Cross-modal interaction*``
**(*NIC_arXiv2014*) Show and Tell: A Neural Image Caption Generator.**<br>
*Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan.*<br>
[[paper]](https://arxiv.org/pdf/1411.4555)

**(*m-RNN_ICLR2015*) Deep Captioning with Multimodal Recurrent Neural Network(M-RNN).**<br>
*Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Zhiheng Huang, Alan Yuille.*<br>
[[paper]](https://arxiv.org/pdf/1412.6632)
[[code]](https://github.com/mjhucla/mRNN-CR)

**(*DAN_CVPR2017*) Dual Attention Networks for Multimodal Reasoning and Matching.**<br>
*Hyeonseob Nam, Jung-Woo Ha, Jeonghee Kim.*<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Nam_Dual_Attention_Networks_CVPR_2017_paper.pdf)

**(*sm-LSTM_CVPR2017*) Instance-aware Image and Sentence Matching with Selective Multimodal LSTM.**<br>
*Yan Huang, Wei Wang, Liang Wang.*<br>
[[paper]](https://arxiv.org/pdf/1611.05588)

**(*CITE_ECCV2018*) Conditional Image-Text Embedding Networks.**<br>
*Bryan A. Plummer, Paige Kordas, M. Hadi Kiapour, Shuai Zheng, Robinson Piramuthu, Svetlana Lazebnik.*<br>
[[paper]](https://arxiv.org/pdf/1711.08389.pdf)

**(*SCAN_ECCV2018*) Stacked Cross Attention for Image-Text Matching.**<br>
*Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He.*<br>
[[paper]](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.pdf)
[[code]](https://github.com/kuanghuei/SCAN)

### ``*Similarity measurement*``
**(*Order-emb_ICLR2016*) Order-Embeddings of Images and Language.**<br>
*Ivan Vendrov, Ryan Kiros, Sanja Fidler, Raquel Urtasun.*<br>
[[paper]](https://arxiv.org/pdf/1511.06361.pdf)

### ``*Loss function*``
**(*VSE++_BMVC2018*) VSE++: Improving Visual-Semantic Embeddings with Hard Negatives.**<br>
*Fartash Faghri, David J. Fleet, Jamie Ryan Kiros, Sanja Fidler.*<br>
[[paper]](https://arxiv.org/pdf/1707.05612.pdf)
[[code]](https://github.com/fartashf/vsepp)

**(*CMPL_ECCV2018*) Deep Cross-Modal Projection Learning for Image-Text Matching.**<br>
*Ying Zhang, Huchuan Lu.*<br>
[[paper]](https://drive.google.com/file/d/1aiBuE1NjW83PGgYbP0eQDGEKr4fqMA6J/view)
[[code]](https://github.com/YingZhangDUT/Cross-Modal-Projection-Learning)

**(*kNN-loss_ACLws2019*) A Strong and Robust Baseline for Text-Image Matching.**<br>
*Fangyu Liu, Rongtian Ye.*<br> 
[[paper]](https://www.aclweb.org/anthology/P19-2023.pdf)

**(*Dual-Path_TOMM2020*) Dual-path Convolutional Image-Text Embeddings with Instance Loss.**<br>
*Zhedong Zheng, Liang Zheng, Michael Garrett, Yi Yang, Mingliang Xu, YiDong Shen.*<br>
[[paper]](https://arxiv.org/pdf/1711.05535)
[[code]](https://github.com/layumi/Image-Text-Embedding)

### ``*Un-supervised or Semi-supervised*``
**(*VSA-AE-MMD_ECCV2018*) Visual-Semantic Alignment Across Domains Using a Semi-Supervised Approach.**<br>
*Angelo Carraggi, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara.*<br>
[[paper]](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Carraggi_Visual-Semantic_Alignment_Across_Domains_Using_a_Semi-Supervised_Approach_ECCVW_2018_paper.pdf)

### ``*Zero-shot or Fewer-shot*``
**(*DEM_CVPR2017*) Learning a Deep Embedding Model for Zero-Shot Learning.**<br>
*Li Zhang, Tao Xiang, Shaogang Gong.*<br>
[[paper]](https://arxiv.org/pdf/1611.05088.pdf)
[[code]](https://github.com/lzrobots/DeepEmbeddingModel_ZSL)

### ``*Adversarial learning*``
**(*CAS_COLING2018*) Learning Visually-Grounded Semantics from Contrastive Adversarial Samples.**<br>
*Haoyue Shi, Jiayuan Mao, Tete Xiao, Yuning Jiang, Jian Sun.*<br>
[[paper]](https://aclweb.org/anthology/C18-1315)
[[code]](https://github.com/ExplorerFreda/VSE-C)

**(*GXN_CVPR2018*) Look, Imagine and Match: Improving Textual-Visual Cross-Modal Retrieval with Generative Models.**<br>
*Jiuxiang Gu, Jianfei Cai, Shafiq Joty, Li Niu, Gang Wang.*<br>
[[paper]](https://arxiv.org/pdf/1711.06420)

**(*TIMAM_ICCV2019*) Adversarial Representation Learning for Text-to-Image Matching.**<br>
*Nikolaos Sarafianos, Xiang Xu, Ioannis A. Kakadiaris.*<br>
[[paper]](https://arxiv.org/pdf/1908.10534.pdf)

### ``*Identification learning*``
**(*LSTM-Q+I_ICCV2015*) VQA: Visual question answering.**<br>
*Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, MargaretMitchell, Dhruv Batra, C Lawrence Zitnick, Devi Parikh.*<br>
[[paper]](http://scholar.google.com.hk/scholar_url?url=http://openaccess.thecvf.com/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf&hl=zh-CN&sa=X&ei=EDHkX9aDAY6CywTJ6a2ACw&scisig=AAGBfm2VHgUhZ4sZPI-ODBqcEdCd34_V8w&nossl=1&oi=scholarr)

**(*Word-NN_CVPR2016*) Learning Deep Representations of Fine-grained Visual Descriptions.**<br>
*Scott Reed, Zeynep Akata, Bernt Schiele, Honglak Lee.*<br>
[[paper]](https://arxiv.org/pdf/1605.05395)

**(*GNA-RNN_CVPR2017*) Person search with natural language description.**<br>
*huang  Li, Tong Xiao, Hongsheng Li, Bolei Zhou, DayuYue, Xiaogang Wang.*<br>
[[paper]](https://arxiv.org/pdf/1702.05729)
[[code]](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)

**(*IATV_ICCV2017*) Identity-aware textual-visual matching with latent co-attention.**<br>
*Shuang Li, Tong Xiao, Hongsheng Li, Wei Yang, Xiaogang Wang.*<br>
[[paper]](https://arxiv.org/pdf/1708.01988)

**(*PWM-ATH_WACV2018*) Improving text-based person search by spatial matching and adaptive threshold.**<br>
*Tianlang Chen, Chenliang Xu, Jiebo Luo.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8354312)

**(*GLA_ECCV2018*) Improving deep visual representation for person re-identification by global and local image-language association.**<br>
*Dapeng Chen, Hongsheng Li, Xihui Liu, Yantao Shen, JingShao, Zejian Yuan, Xiaogang Wang.*<br>
[[paper]](https://arxiv.org/pdf/1808.01571)

**(*PMA_AAAI2020*) Pose-Guided Multi-Granularity Attention Network for Text-Based Person Search.**<br>
*Ya Jing, Chenyang Si, Junbo Wang, Wei Wang, Liang Wang, Tieniu Tan.*<br>
[[paper]](https://arxiv.org/pdf/1809.08440)

### ``*Related works*``
**(*Word2Vec_NIPS2013*) Distributed Representations of Words and Phrases and their Compositionality.**<br>
*Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean.*<br>
[[paper]](https://arxiv.org/pdf/1310.4546)

**(*DVSQ_CVPR2017*) Deep Visual-Semantic Quantization for Efficient Image Retrieval.**<br>
*Yue Cao, Mingsheng Long, Jianmin Wang, Shichen Liu.*<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Cao_Deep_Visual-Semantic_Quantization_CVPR_2017_paper.pdf)

**(*VSE-ens_AAAI2018*) VSE-ens: Visual-Semantic Embeddings with Efficient Negative Sampling.**<br>
*Guibing Guo, Songlin Zhai, Fajie Yuan, Yuan Liu, Xingwei Wang.*<br>
[[paper]](https://arxiv.org/pdf/1801.01632.pdf)

**(*ILU_ACL2018*) Illustrative Language Understanding: Large-Scale Visual Grounding with Image Search.**<br>
*Jamie Kiros, William Chan, Geoffrey Hinton.*<br>
[[paper]](https://aclweb.org/anthology/P18-1085)

**(*HTG_ECCV2018*) An Adversarial Approach to Hard Triplet Generation.**<br>
*Yiru Zhao, Zhongming Jin, Guo-jun Qi, Hongtao Lu, Xian-sheng Hua.*<br>
[[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yiru_Zhao_A_Principled_Approach_ECCV_2018_paper.pdf)

**(*BUTD_CVPR2018*) Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering.**<br>
*Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, Lei Zhang.*<br>
[[paper]](https://arxiv.org/pdf/1707.07998)

**(*WebNet_ECCV2018*) CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images.**<br>
*Sheng Guo, Weilin Huang, Haozhi Zhang, Chenfan Zhuang, Dengke Dong, Matthew R. Scott, Dinglong Huang.*<br>
[[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sheng_Guo_CurriculumNet_Learning_from_ECCV_2018_paper.pdf)
[[code]](https://github.com/MalongTech/research-curriculumnet)

**(*DML_CVPR2018*) Deep Mutual Learning.**<br>
*Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu.*<br>
[[paper]](https://drive.google.com/file/d/1Jr1uWF3RImqNRsDMKTJVIswUVfMKYnuE/view)
[[code]](https://github.com/YingZhangDUT/Deep-Mutual-Learning)

### *Posted in*


-----------------------------------------------------------
**Unified Visual-Semantic Embeddings: Bridging Vision and Language with Structured Meaning Representations.**<br>
*Hao Wu, Jiayuan Mao, Yufeng Zhang, Yuning Jiang, Lei Li, Weiwei Sun, Wei-Ying Ma.*<br>
**_(CVPR 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1904.05521.pdf)

**Engaging Image Captioning via Personality.**<br>
*Kurt Shuster, Samuel Humeau, Hexiang Hu, Antoine Bordes, Jason Weston.*<br>
**_(CVPR 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1810.10665.pdf)

**Polysemous Visual-Semantic Embedding for Cross-Modal Retrieval.**<br>
*Yale Song, Mohammad Soleymani.*<br>
**_(CVPR 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1906.04402.pdf)
 
**Composing Text and Image for Image Retrieval - An Empirical Odyssey.**<br>
*Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.*<br>
**_(CVPR 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1812.07119.pdf)

**Annotation Efficient Cross-Modal Retrieval with Adversarial Attentive Alignment.**<br>
*Po-Yao Huang, Guoliang Kang, Wenhe Liu, Xiaojun Chang, Alexander G Hauptmann.*<br>
**_(ACM MM 2019)_**<br>
[[paper]](http://www.cs.cmu.edu/~poyaoh/data/ann.pdf)

**Visual Semantic Reasoning for Image-Text Matching.**<br>
*Kunpeng Li, Yulun Zhang, Kai Li, Yuanyuan Li, Yun Fu.*<br>
**_(ICCV 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1909.02701.pdf)

**CAMP: Cross-Modal Adaptive Message Passing for Text-Image Retrieval.**<br>
*Zihao Wang, Xihui Liu, Hongsheng Li, Lu Sheng, Junjie Yan, Xiaogang Wang, Jing Shao.*<br>
**_(ICCV 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1909.05506.pdf)

**Align2Ground: Weakly Supervised Phrase Grounding Guided by Image-Caption Alignment.**<br>
*Samyak Datta, Karan Sikka, Anirban Roy, Karuna Ahuja, Devi Parikh, Ajay Divakaran.*<br>
**_(ICCV 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1903.11649.pdf)

**Multi-Head Attention with Diversity for Learning Grounded Multilingual Multimodal Representations.**<br>
*Po-Yao Huang, Xiaojun Chang, Alexander Hauptmann.*<br>
**_(EMNLP 2019)_**<br>
[[paper]](https://www.aclweb.org/anthology/D19-1154.pdf)

**Unsupervised Discovery of Multimodal Links in Multi-Image, Multi-Sentence Documents.**<br>
*Jack Hessel, Lillian Lee, David Mimno.*<br>
**_(EMNLP 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1904.07826.pdf)
[[code]](https://github.com/jmhessel/multi-retrieval)

**HAL: Improved Text-Image Matching by Mitigating Visual Semantic Hubs.**<br>
*Fangyu Liu, Rongtian Ye, Xun Wang, Shuaipeng Li.*<br>
**_(AAAI 2020)_**<br>
[[paper]](https://arxiv.org/pdf/1911.10097v1.pdf)
[[code]](https://github.com/hardyqr/HAL) (PyTorch)

**Ladder Loss for Coherent Visual-Semantic Embedding.**<br>
*Mo Zhou, Zhenxing Niu, Le Wang, Zhanning Gao, Qilin Zhang, Gang Hua.*<br>
**_(AAAI 2020)_**<br>
[[paper]](https://arxiv.org/pdf/1911.07528.pdf)

**Expressing Objects just like Words: Recurrent Visual Embedding for Image-Text Matching.** <br>
*Tianlang Chen, Jiebo Luo.*<br>
**_(AAAI 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2002.08510.pdf)

**Adaptive Cross-modal Embeddings for Image-Text Alignment.**<br>
*Jonatas Wehrmann, Camila Kolling, Rodrigo C Barros.*<br>
**_(AAAI 2020)_**<br>
[[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/6915)
[[code]](https://github.com/jwehrmann/retrieval.pytorch) (PyTorch)

**Graph Structured Network for Image-Text Matching**.<br>
*Chunxiao Liu, Zhendong Mao, Tianzhu Zhang, Hongtao Xie, Bin Wang, Yongdong Zhang.*<br>
**_(CVPR 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2004.00277.pdf)

**IMRAM: Iterative Matching with Recurrent Attention Memory for Cross-Modal Image-Text Retrieval.**<br>
*Hui Chen, Guiguang Ding, Xudong Liu, Zijia Lin, Ji Liu, Jungong Han.*<br>
**_(CVPR 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2003.03772.pdf)

**Visual-Semantic Matching by Exploring High-Order Attention and Distraction.**<br>
*Yongzhi Li, Duo Zhang, Yadong Mu.*<br>
**_(CVPR 2020)_**<br>
[[paper]](https://pkumyd.github.io/paper/CVPR2020_LYZ.pdf)

**Multi-Modality Cross Attention Network for Image and Sentence Matching.**<br>
*Xi Wei, Tianzhu Zhang, Yan Li, Yongdong Zhang, Feng Wu.*<br>
**_(CVPR 2020)_**<br>
[[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Multi-Modality_Cross_Attention_Network_for_Image_and_Sentence_Matching_CVPR_2020_paper.pdf)

**Context-Aware Attention Network for Image-Text Retrieval.**<br>
*Qi Zhang, Zhen Lei, Zhaoxiang Zhang, Stan Z. Li.*<br>
**_(CVPR 2020)_**<br>
[[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Context-Aware_Attention_Network_for_Image-Text_Retrieval_CVPR_2020_paper.pdf)

**Universal Weighting Metric Learning for Cross-Modal Matching.**<br>
*Jiwei Wei, Xing Xu, Yang Yang, Yanli Ji, Zheng Wang, Heng Tao Shen.*<br>
**_(CVPR 2020)_**<br>
[[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Universal_Weighting_Metric_Learning_for_Cross-Modal_Matching_CVPR_2020_paper.pdf)

**Graph Optimal Transport for Cross-Domain Alignment.**<br>
*Liqun Chen, Zhe Gan, Yu Cheng, Linjie Li, Lawrence Carin, Jingjing Liu.*<br>
**_(ICML 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2006.14744.pdf)

**Adaptive Offline Quintuplet Loss for Image-Text Matching.**<br>
*Tianlang Chen, Jiajun Deng, Jiebo Luo.*<br>
**_(ECCV 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2003.03669.pdf)
[[code]](https://github.com/sunnychencool/AOQ)(PyTorch)

**Learning Joint Visual Semantic Matching Embeddings for Language-guided Retrieval.**<br>
*Yanbei Chen, Loris Bazzani.*<br>
**_(ECCV 2020)_**<br>
[[paper]](https://assets.amazon.science/5b/db/440af26349adb83c77c85cd11922/learning-joint-visual-semantic-matching-embeddings-for-text-guided-retrieval.pdf)

**Consensus-Aware Visual-Semantic Embeddingfor Image-Text Matching.**<br>
*Haoran Wang, Ying Zhang, Zhong Ji, Yanwei Pang, Lin Ma.*<br>
**_(ECCV 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2007.08883.pdf)
[[code]](https://github.com/BruceW91/CVSE)(PyTorch)

**Contrastive Learning for Weakly Supervised Phrase Grounding.**<br>
*Tanmay Gupta, Arash Vahdat, Gal Chechik, Xiaodong Yang, Jan Kautz, Derek Hoiem.*<br>
**_(ECCV 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2006.09920.pdf)
[[code]](https://github.com/BigRedT/info-ground)(PyTorch)

**Preserving Semantic Neighborhoods for Robust Cross-modal Retrieval.**<br>
*Christopher Thomas, Adriana Kovashka.*<br>
**_(ECCV 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2007.08617.pdf)


**Probing Multimodal Embeddings for Linguistic Properties: the Visual-Semantic Case.**<br>
*Adam Dahlgren Lindström, Suna Bensch, Johanna Björklund, Frank Drewes.*<br>
**_(COLING 2020)_**<br>
[[paper]](https://www.aclweb.org/anthology/2020.coling-main.64.pdf)
[[code]](https://github.com/dali-does/vse-probing)(PyTorch)

**Large scale image annotation: learning to rank with joint word-image embeddings.**<br>
*Jason Weston, Samy Bengio, Nicolas Usunier.*<br>
**_(Machine Learning 2010)_**<br>
[[paper]](https://link.springer.com/content/pdf/10.1007%2Fs10994-010-5198-3.pdf)

**Learning Two-Branch Neural Networks for Image-Text Matching Tasks.**<br>
*Liwei Wang, Yin Li, Jing Huang, Svetlana Lazebnik.*<br>
**_(IPAMI 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1704.03470.pdf)
[[code]](https://github.com/lwwang/Two_branch_network)(TF)

**Upgrading the Newsroom: An Automated Image Selection System for News Articles.**<br>
*Fangyu Liu, Rémi Lebret, Didier Orel, Philippe Sordet, Karl Aberer.*<br>
**_(ACM TOMM 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2004.11449.pdf)
[[slides]](http://fangyuliu.me/media/others/lsir_talk_final_version_0.3.pdf)
[[demo]](https://modemos.epfl.ch/article)
