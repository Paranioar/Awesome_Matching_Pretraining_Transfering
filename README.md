Cross-modal_Retrieval_Tutorial
==============================
The Tutorial of Image-Text Matching for Preliminary Insight.  
Some state-of-the-arts are temporarily stored in [Posted in](#posted-in). The tutorial will be constantly updated.


## ``Catalogue ``
* [Peformance Comparison](#peformance-comparison)
    * [Flickr8K](#performance-of-flickr8k)
    * [Flickr30K](#performance-of-flickr30k)
    * [MSCOCO1K](#performance-of-mscoco1k)
    * [MSCOCO5K](#performance-of-mscoco5k)
    * [CUHK-PEDES](#performance-of-cuhk-pedes)
    * [CUB-Flowers](#performance-of-cub-flowers)

* [Methods Summary](#method-summary)
    * [Generic-Feature Extraction](#generic-feature-extraction)
    * [Cross-Modal Interaction](#cross-modal-interaction)
    * [Similarity Measurement](#similarity-measurement)
    * [Loss Function](#loss-function)
    * [Un-Supervised or Semi-Supervised](#un-supervised-or-semi-supervised)
    * [Zero-Shot or Fewer-Shot](#zero-shot-or-fewer-shot)
    * [Adversarial Learning](#adversarial-learning)
    * [Commonsense Learning](#commonsense-learning)
    * [Identification Learning](#identification-learning)
    * [Scene-Text Learning](#scene-text-learning)
    * [Related Works](#related-works)  
    * [Posted in](#posted-in)
* [Other Resources](#other-resources)  
    * [Fewshot Learning](#fewshot-learning)
    * [Graph Learning](#graph-learning)
    * [Transformer Learning](#transformer-learning)
    

## ``Peformance Comparison``

### *Performance of Flickr8K*
**(*\** indicates Ensemble models, *^* indicates questionable authen)**
<table>
   <tr> <td rowspan="2">Method_name</td> <td rowspan="2", align="center">Concise_note</td> 
        <td colspan="3", align="center">Sentence retrieval</td> <td colspan="3", align="center">Image retrieval</td> </tr>
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
   <tr> <td>SPE</td><td>VggNet</td> <td>30.1</td><td>60.4</td><td>73.7</td> <td>23.0</td><td>51.3</td><td>64.8</td> </tr>
   <tr> <td>FV</td><td>GMM+HGLMM</td> <td>31.0</td><td>59.3</td><td>73.7</td> <td>21.2</td><td>50.0</td><td>64.8</td> </tr>
   <tr> <td>NAA</td><td>ResNet</td> <td>37.2</td><td>68.1</td><td>79.1</td> <td>27.7</td><td>59.6</td><td>71.8</td> </tr>
   <tr> <td>SCAN*</td><td>BUTD</td> <td>52.2</td><td>81.0</td><td>89.2</td> <td>38.3</td><td>67.8</td><td>78.9</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Image</td> <td>48.5</td><td>78.1</td><td>85.3</td> <td>32.0</td><td>61.4</td><td>73.9</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Text</td> <td>52.1</td><td>81.5</td><td>90.1</td> <td>40.2</td><td>69.0</td><td>79.2</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Full</td> <td>54.7</td><td>84.2</td><td>91.0</td> <td>41.0</td><td>69.2</td><td>79.9</td> </tr>
</table> 

### *Performance of Flickr30K*
<table>
   <tr> <td rowspan="2">Method_name</td> <td rowspan="2", align="center">Concise_note</td>
        <td colspan="3", align="center">Sentence retrieval</td> <td colspan="3", align="center">Image retrieval</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td> <td>R@1</td><td>R@5</td><td>R@10</td> </tr>
   <tr> <td>DeViSE</td><td>RCNN</td> <td>4.5</td><td>18.1</td><td>29.2</td> <td>6.7</td><td>21.9</td><td>32.7</td> </tr>
   <tr> <td>SDT-RNN</td><td>RCNN</td> <td>9.6</td><td>29.8</td><td>41.1</td> <td>8.9</td><td>29.8</td><td>41.1</td> </tr>
   <tr> <td>DeFrag</td><td>RCNN</td> <td>14.2</td><td>37.7</td><td>51.3</td> <td>10.2</td><td>30.8</td><td>44.2</td> </tr>
   <tr> <td>DeFrag</td><td>ftRCNN</td> <td>16.4</td><td>40.2</td><td>54.7</td> <td>10.3</td><td>31.4</td><td>44.5</td> </tr>
   <tr> <td>DCCA</td><td>AlexNet</td> <td>16.7</td><td>39.3</td><td>52.9</td> <td>12.6</td><td>31.0</td><td>43.0</td> </tr>
   <tr> <td>NIC</td><td>GoogleNet</td> <td>17</td><td>--</td><td>56</td> <td>17</td><td>--</td><td>57</td> </tr>
   <tr> <td>DVSA</td><td>DepTree</td> <td>20.0</td><td>46.6</td><td>59.4</td> <td>15.0</td><td>36.5</td><td>48.2</td> </tr>
   <tr> <td>DVSA</td><td>RCNN</td> <td>22.2</td><td>48.2</td><td>61.4</td> <td>15.2</td><td>37.7</td><td>50.5</td> </tr>
   <tr> <td>UVSE</td><td>AlexNet</td> <td>14.8</td><td>39.2</td><td>50.9</td> <td>11.8</td><td>34.0</td><td>46.3</td> </tr>
   <tr> <td>UVSE</td><td>VggNet</td> <td>23.0</td><td>50.7</td><td>62.9</td> <td>16.8</td><td>42.0</td><td>56.5</td> </tr>
   <tr> <td>LRCN</td><td>VggNet</td> <td>23.6</td><td>46.6</td><td>58.3</td> <td>17.5</td><td>40.3</td><td>50.8</td> </tr>
   <tr> <td>m-CNN*</td><td>OverFeat</td> <td>20.1</td><td>44.2</td><td>56.3</td> <td>15.9</td><td>40.3</td><td>51.9</td> </tr>
   <tr> <td>m-CNN*</td><td>VggNet</td> <td>33.6</td><td>64.1</td><td>74.9</td> <td>26.2</td><td>56.3</td><td>69.6</td> </tr>
   <tr> <td>m-RNN</td><td>AlexNet</td> <td>18.4</td><td>40.2</td><td>50.9</td> <td>12.6</td><td>31.2</td><td>41.5</td> </tr>
   <tr> <td>m-RNN</td><td>VggNet</td> <td>35.4</td><td>63.8</td><td>73.7</td> <td>22.8</td><td>50.7</td><td>63.1</td> </tr>
   <tr> <td>FV</td><td>GMM+HGLMM</td> <td>35.0</td><td>62.0</td><td>73.8</td> <td>25.0</td><td>52.7</td><td>66.0</td> </tr>
   <tr> <td>HM-LSTM</td><td>RCNN</td> <td>38.1</td><td>--</td><td>76.5</td> <td>27.7</td><td>--</td><td>68.8</td> </tr>
   <tr> <td>SPE</td><td>VggNet</td> <td>40.3</td><td>68.9</td><td>79.9</td> <td>29.7</td><td>60.1</td><td>72.1</td> </tr>
   <tr> <td>sm-LSTM</td><td>VggNet</td> <td>42.4</td><td>67.5</td><td>79.9</td> <td>28.2</td><td>57.0</td><td>68.4</td> </tr>
   <tr> <td>sm-LSTM*</td><td>VggNet</td> <td>42.5</td><td>71.9</td><td>81.5</td> <td>30.2</td><td>60.4 </td><td>72.3</td> </tr>
   <tr> <td>CSE</td><td>ResNet</td> <td>44.6</td><td>74.3</td><td>83.8</td> <td>36.9</td><td>69.1</td><td>79.6</td> </tr>
   <tr> <td>RRF-Net</td><td>ResNet</td> <td>47.6</td><td>77.4</td><td>87.1</td> <td>35.4</td><td>68.3</td><td>79.9</td> </tr>
   <tr> <td>CMPL</td><td>MobileNet</td> <td>40.3</td><td>66.9</td><td>76.7</td> <td>30.4</td><td>58.2</td><td>68.5</td> </tr>
   <tr> <td>CMPL</td><td>ResNet</td> <td>49.6</td><td>76.8</td><td>86.1</td> <td>37.3</td><td>65.7</td><td>75.5</td> </tr>
   <tr> <td>2WayNet</td><td>VggNet</td> <td>49.8</td><td>67.5</td><td>--</td> <td>36.0</td><td>55.6</td><td>--</td> </tr>
   <tr> <td>VSE++</td><td>VggNet</td> <td>41.3</td><td>69.1</td><td>77.9</td> <td>31.4</td><td>60.0</td><td>71.2</td> </tr>
   <tr> <td>VSE++</td><td>ResNet</td> <td>52.9</td><td>80.5</td><td>87.2</td> <td>39.6</td><td>70.1</td><td>79.5</td> </tr>
   <tr> <td>TIMAM</td><td>ResNet, Bert</td> <td>53.1</td><td>78.8</td><td>87.6</td> <td>42.6</td><td>71.6</td><td>81.9</td> </tr>
   <tr> <td>DAN</td><td>VggNet</td> <td>41.4</td><td>73.5</td><td>82.5</td> <td>31.8</td><td>61.7</td><td>72.5</td> </tr>
   <tr> <td>DAN</td><td>ResNet</td> <td>55.0</td><td>81.8</td><td> 89.0</td> <td>39.4</td><td>69.2</td><td>79.1</td> </tr>
   <tr> <td>NAA</td><td>ResNet</td> <td>55.1</td><td>80.3</td><td>89.6</td> <td>39.4</td><td>68.8</td><td>79.9</td> </tr>
   <tr> <td>SCO</td><td>VggNet</td> <td>44.2</td><td>74.1</td><td>83.6</td> <td>32.8</td><td>64.3</td><td>74.9</td> </tr>
   <tr> <td>SCO</td><td>ResNet</td> <td>55.5</td><td>82.0</td><td>89.3</td> <td>41.1</td><td>70.5</td><td>80.1</td> </tr>
   <tr> <td>Dual-Path</td><td>VggNet</td> <td>47.6</td><td>77.3</td><td>87.1</td> <td>35.3</td><td>66.6</td><td>78.2</td> </tr>
   <tr> <td>Dual-Path</td><td>ResNet</td> <td>55.6</td><td>81.9</td><td>89.5</td> <td>39.1</td><td>69.2</td><td>80.9</td> </tr>
   <tr> <td>CVSE++</td><td>ResNet</td> <td>56.6</td><td>82.5</td><td>90.2</td> <td>42.4</td><td>71.6</td><td>80.8</td> </tr>
   <tr> <td>GXN</td><td>ResNet</td> <td>56.8</td><td>--</td><td>89.6</td> <td>41.5</td><td>--</td><td>80.1</td> </tr>
   <tr> <td>Align2Ground</td><td>BUTD</td> <td>--</td><td>--</td><td>--</td> <td>49.7</td><td>74.8</td><td>83.3</td> </tr>
   <tr> <td>A3VSE</td><td>BUTD</td> <td>65.0</td><td>89.2</td><td>94.5</td> <td>49.5</td><td>79.5</td><td>86.6</td> </tr>
   <tr> <td>MTFN</td><td>BUTD</td> <td>63.1</td><td>85.8</td><td>92.4</td> <td>46.3</td><td>75.3</td><td>83.6</td> </tr>
   <tr> <td>MTFN</td><td>BUTD, RR_no_STT</td> <td>65.3</td><td>88.3</td><td>93.3</td> <td>46.7</td><td>75.9</td><td>83.8</td> </tr>
   <tr> <td>MTFN</td><td>BUTD, RR_STT</td> <td>65.3</td><td>88.3</td><td>93.3</td> <td>52.0</td><td>80.1</td><td>86.1</td> </tr>
   <tr> <td>R-SCAN</td><td>BUTD, VrR-VG</td> <td>66.3</td><td>90.6</td><td>96.0</td> <td>51.4</td><td>77.8</td><td>84.9</td> </tr>
   <tr> <td>SAVE</td><td>ResNet</td> <td>67.2</td><td>88.3</td><td>94.2</td> <td>49.8</td><td>78.7</td><td>86.2</td> </tr>
   <tr> <td>SCAN</td><td>BUTD, t2i_AVE</td> <td>61.8</td><td>87.5</td><td>93.7</td> <td>45.8</td><td>74.4</td><td>83.0</td> </tr>
   <tr> <td>SCAN</td><td>BUTD, i2t_AVE</td> <td>67.9</td><td>89.0</td><td>94.4</td> <td>43.9</td><td>74.2</td><td>82.8</td> </tr>
   <tr> <td>SCAN*</td><td>BUTD, AVE+LSE</td> <td>67.4</td><td>90.3</td><td>95.8</td> <td>48.6</td><td>77.7</td><td>85.2</td> </tr>
   <tr> <td>BFAN</td><td>BUTD, prob</td> <td>65.5</td><td>89.4</td><td>--</td> <td>47.9</td><td>77.6</td><td>--</td> </tr>
   <tr> <td>BFAN</td><td>BUTD, equal</td> <td>64.5</td><td>89.7</td><td>--</td> <td>48.8</td><td>77.3</td><td>--</td> </tr>
   <tr> <td>BFAN*</td><td>BUTD</td> <td>68.1</td><td>91.4</td><td>--</td> <td>50.8</td><td>78.4</td><td>--</td> </tr>
   <tr> <td>CAMP</td><td>BUTD</td> <td>68.1</td><td>89.7</td><td>95.2</td> <td>51.5</td><td>77.1</td><td>85.3</td> </tr>
   <tr> <td>RDAN</td><td>BUTD</td> <td>68.1</td><td>91.0</td><td>95.9</td> <td>54.1</td><td>80.9</td><td>87.2</td> </tr>
   <tr> <td>GSLS</td><td>ResNet, BUTD</td> <td>68.2</td><td>89.1</td><td>94.5</td> <td>43.4</td><td>73.5</td><td>82.5</td> </tr>
   <tr> <td>Personality</td><td>ResNeXt, Transformer</td> <td>68.4</td><td>90.6</td><td>95.3</td> <td>--</td><td>--</td><td>--</td> </tr>
   <tr> <td>CASC</td><td>ResNet</td> <td>68.5</td><td>90.6</td><td>95.9</td> <td>50.2</td><td>78.3</td><td>86.3</td> </tr>
   <tr> <td>GVSE*</td><td>BUTD</td> <td>68.5</td><td>90.9</td><td>95.5</td> <td>50.6</td><td>79.8</td><td>87.6</td> </tr>
   <tr> <td>HAL</td><td>SCAN_i2t</td> <td>68.6</td><td>89.9</td><td>94.7</td> <td>46.0</td><td>74.0</td><td>82.3</td> </tr>
   <tr> <td>OAN</td><td>BUTD</td> <td>68.6</td><td>93.0</td><td>96.0</td> <td>53.3</td><td>80.1</td><td>87.1</td> </tr>
   <tr> <td>SAEM</td><td>BUTD, Bert</td> <td>69.1</td><td>91.0</td><td>95.1</td> <td>52.4</td><td>81.1</td><td>88.1</td> </tr>
   <tr> <td>MPL</td><td>SCAN_i2t</td> <td>69.4</td><td>89.9</td><td>95.4</td> <td>47.5</td><td>75.5</td><td>83.1</td> </tr>
   <tr> <td>PFAN</td><td>BUTD, t2i</td> <td>66.0</td><td>89.6</td><td>94.3</td> <td>49.6</td><td>77.0</td><td>84.2</td> </tr>
   <tr> <td>PFAN</td><td>BUTD, i2t</td> <td>67.6</td><td>90.0</td><td>93.8</td> <td>45.7</td><td>74.7</td><td>83.6</td> </tr>
   <tr> <td>PFAN*</td><td>BUTD</td> <td>70.0</td><td>91.8</td><td>95.0</td> <td>50.4</td><td>78.7</td><td>86.1</td> </tr>
   <tr> <td>CAAN</td><td>BUTD</td> <td>70.1</td><td>91.6</td><td>97.2</td> <td>52.8</td><td>79.0</td><td>87.9</td> </tr>
   <tr> <td>DP-RNN</td><td>BUTD</td> <td>70.2</td><td>91.6</td><td>95.8</td> <td>55.5</td><td>81.3</td><td>88.2</td> </tr>
   <tr> <td>HOAD</td><td>BUTD</td> <td>70.8</td><td>92.7</td><td>96.0</td> <td>59.5</td><td>85.6</td><td>91.0</td> </tr>
   <tr> <td>HOAD</td><td>BUTD, +Dist</td> <td>70.8</td><td>92.7</td><td>96.0</td> <td>60.9</td><td>86.1</td><td>91.0</td> </tr>
   <tr> <td>GOT</td><td>SCAN_i2t</td> <td>70.9</td><td>92.8</td><td>95.5</td> <td>50.7</td><td>78.7</td><td>86.2</td> </tr>
   <tr> <td>VSRN*</td><td>BUTD</td> <td>71.3</td><td>90.6</td><td>96.0</td> <td>54.7</td><td>81.8</td><td>88.2</td> </tr>
   <tr> <td>SCG</td><td>VggNet, Prod</td> <td>57.2</td><td>85.1</td><td>92.1</td> <td>40.1</td><td>69.5</td><td>79.5</td> </tr>
   <tr> <td>SCG</td><td>VggNet, Gated</td> <td>71.8</td><td>90.8</td><td>94.8</td> <td>49.3</td><td>76.4</td><td>85.6</td> </tr>
   <tr> <td>SGM</td><td>BUTD</td> <td>71.8</td><td>91.7</td><td>95.5</td> <td>53.5</td><td>79.6</td><td>86.5</td> </tr>
   <tr> <td>ADDR*</td><td>BUTD, BFAN</td> <td>71.3</td><td>91.5</td><td>96.4</td> <td>54.0</td><td>80.0</td><td>87.6</td> </tr>
   <tr> <td>ADDR*</td><td>BUTD, SCAN</td> <td>72.1</td><td>93.1</td><td>96.1</td> <td>53.5</td><td>80.4</td><td>87.4</td> </tr>
   <tr> <td>ADDR*</td><td>BUTD, VSRN</td> <td>73.0</td><td>92.5</td><td>96.6</td> <td>55.6</td><td>82.0</td><td>88.9</td> </tr>
   <tr> <td>AOQ*</td><td>BUTD, SCAN</td> <td>70.3</td><td>92.0</td><td>95.5</td> <td>50.0</td><td>79.2</td><td>86.2</td> </tr>
   <tr> <td>AOQ*</td><td>BUTD, VSRN</td> <td>72.8</td><td>91.8</td><td>95.8</td> <td>55.3</td><td>82.2</td><td>88.4</td> </tr>
   <tr> <td>AOQ*</td><td>BUTD, BFAN</td> <td>73.2</td><td>94.5</td><td>97.0</td> <td>54.0</td><td>80.3</td><td>87.7</td> </tr>
   <tr> <td>CVSE^</td><td>BUTD</td> <td>73.5</td><td>92.1</td><td>95.8</td> <td>52.9</td><td>80.4</td><td>87.8</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Image</td> <td>67.0</td><td>90.5</td><td>95.6</td> <td>51.2</td><td>78.2</td><td>85.5</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Text</td> <td>68.8</td><td>91.6</td><td>96.0</td> <td>53.0</td><td>79.0</td><td>87.1</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Full</td> <td>74.1</td><td>93.0</td><td>96.6</td> <td>53.9</td><td>79.4</td><td>87.2</td> </tr>
   <tr> <td>MMCA</td><td>BUTD, Bert</td> <td>74.2</td><td>92.8</td><td>96.4</td> <td>54.8</td><td>81.4</td><td>87.8</td> </tr>
   <tr> <td>SAN^</td><td>VggNet</td> <td>67.0</td><td>88.0</td><td>94.6</td> <td>51.4</td><td>77.2</td><td>85.2</td> </tr>
   <tr> <td>SAN^</td><td>ResNet</td> <td>75.5</td><td>92.6</td><td>96.2</td> <td>60.1</td><td>84.7</td><td>90.6</td> </tr>
   <tr> <td>GSMN</td><td>BUTD, sparse</td> <td>71.4</td><td>92.0</td><td>96.1</td> <td>53.9</td><td>79.7</td><td>87.1</td> </tr>
   <tr> <td>GSMN</td><td>BUTD, dense</td> <td>72.6</td><td>93.5</td><td>96.8</td> <td>53.7</td><td>80.0</td><td>87.0</td> </tr>
   <tr> <td>GSMN*</td><td>BUTD</td> <td>76.4</td><td>94.3</td><td>97.3</td> <td>57.4</td><td>82.3</td><td>89.0</td> </tr>
   <tr> <td>ADAPT</td><td>BUTD, i2t</td> <td>70.2</td><td>90.8</td><td>95.8</td> <td>55.5</td><td>82.7</td><td>89.8</td> </tr>
   <tr> <td>ADAPT</td><td>BUTD, t2i</td> <td>73.6</td><td>93.7</td><td>96.7</td> <td>57.0</td><td>83.6</td><td>90.3</td> </tr>
   <tr> <td>ADAPT*</td><td>BUTD</td> <td>76.6</td><td>95.4</td><td>97.6</td> <td>60.7</td><td>86.6</td><td>92.0</td> </tr>
   <tr> <td>SGRAF</td><td>BUTD, SAF</td> <td>73.7</td><td>93.3</td><td>96.3</td> <td>56.1</td><td>81.5</td><td>88.0</td> </tr>
   <tr> <td>SGRAF</td><td>BUTD, SGR</td> <td>75.2</td><td>93.3</td><td>96.6</td> <td>56.2</td><td>81.0</td><td>86.5</td> </tr>
   <tr> <td>SGRAF*</td><td>BUTD</td> <td>77.8</td><td>94.1</td><td>97.4</td> <td>58.5</td><td>83.0</td><td>88.8</td> </tr>
   <tr> <td>ACMM</td><td>BUTD</td> <td>80.0</td><td>95.5</td><td>98.2</td> <td>50.2</td><td>76.8</td><td>84.7</td> </tr>
   <tr> <td>ACMM*</td><td>BUTD</td> <td>85.2</td><td>96.7</td><td>98.4</td> <td>53.8</td><td>79.8</td><td>86.8</td> </tr>
</table> 

### *Performance of MSCOCO1K*
<table>
   <tr> <td rowspan="2">Method_name</td> <td rowspan="2", align="center">Concise_note</td> 
        <td colspan="3", align="center">Sentence retrieval</td> <td colspan="3", align="center">Image retrieval</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td> <td>R@1</td><td>R@5</td><td>R@10</td> </tr>
   <tr> <td>STV</td><td>combine-skip</td> <td>33.8</td><td>67.7</td><td>82.1</td> <td>25.9</td><td>60.0</td><td>74.6</td> </tr>
   <tr> <td>DVSA</td><td>RCNN</td> <td>38.4</td><td>69.9</td><td>80.5</td> <td>27.4</td><td>60.2</td><td>74.8</td> </tr>
   <tr> <td>FV</td><td>GMM+HGLMM</td> <td>39.4</td><td>67.9</td><td>80.9</td> <td>25.1</td><td>59.8</td><td>76.6</td> </tr>
   <tr> <td>m-RNN</td><td>VggNet</td> <td>41.0</td><td>73.0</td><td>83.5</td> <td>29.0</td><td>42.2</td><td>77.0</td> </tr>
   <tr> <td>m-CNN*</td><td>VggNet</td> <td>42.8</td><td>73.1</td><td>84.1</td> <td>32.6</td><td>68.6</td><td>82.8</td> </tr>
   <tr> <td>UVSE</td><td>VggNet</td> <td>43.4</td><td>75.7</td><td>85.8</td> <td>31.0</td><td>66.7</td><td>79.9</td> </tr>
   <tr> <td>HM-LSTM</td><td>RCNN</td> <td>43.9</td><td>--</td><td>87.8</td> <td>36.1</td><td>--</td><td>86.7</td> </tr>
   <tr> <td>Order-emb</td><td>VggNet</td> <td>46.7</td><td>--</td><td>88.9</td> <td>37.9</td><td>--</td><td>85.9</td> </tr>
   <tr> <td>SPE</td><td>VggNet</td> <td>50.1</td><td>79.7</td><td>89.2</td> <td>39.6</td><td>75.2</td><td>86.9</td> </tr> 
   <tr> <td>SEAM</td><td>VggNet</td> <td>50.7</td><td>81.4</td><td>90.9</td> <td>40.3</td><td>75.7</td><td>87.4</td> </tr>
   <tr> <td>sm-LSTM</td><td>VggNet</td> <td>52.4</td><td>81.7</td><td>90.8</td> <td>38.6</td><td>73.4</td><td>84.6</td> </tr>
   <tr> <td>sm-LSTM*</td><td>VggNet</td> <td>53.2</td><td>83.1</td><td>91.5</td> <td>40.7</td><td>75.8</td><td>87.4</td> </tr>  
   <tr> <td>CMPL</td><td>MobileNet</td> <td>52.9</td><td>83.8</td><td>92.1</td> <td>41.3</td><td>74.6</td><td>85.9</td> </tr>
   <tr> <td>2WayNet</td><td>VggNet</td> <td>55.8</td><td>75.2</td><td>--</td> <td>39.7</td><td>63.3</td><td>--</td> </tr>
   <tr> <td>CMPM</td><td>ResNet</td> <td>56.1</td><td>86.3</td><td>92.9</td> <td>44.6</td><td>78.8</td><td>89.0</td> </tr>
   <tr> <td>CSE</td><td>ResNet</td> <td>56.3</td><td>84.4</td><td>92.2</td> <td>45.7</td><td>81.2</td><td>90.6</td> </tr>    
   <tr> <td>RRF-Net</td><td>ResNet</td> <td>56.4</td><td>85.3</td><td>91.5</td> <td>43.9</td><td>78.1</td><td>88.6</td> </tr>   
   <tr> <td>CHAIN-VSE</td><td>VggNet</td> <td>51.6</td><td>82.0</td><td>91.3</td> <td>38.6</td><td>75.1</td><td>87.2</td> </tr>
   <tr> <td>CHAIN-VSE</td><td>ResNet</td> <td>59.4</td><td>88.0</td><td>94.2</td> <td>43.5</td><td>79.8</td><td>90.2</td> </tr>
   <tr> <td>NAA</td><td>ResNet</td> <td>61.3</td><td>87.9</td><td>95.4</td> <td>47.0</td><td>80.8</td><td>90.1</td> </tr>
   <tr> <td>VSE++</td><td>VggNet</td> <td>57.2</td><td>86.0</td><td>93.3</td> <td>45.9</td><td>79.4</td><td>89.1</td> </tr>
   <tr> <td>VSE++</td><td>ResNet</td> <td>64.6</td><td>90.0</td><td>95.7</td> <td>52.0</td><td>84.3</td><td>92.0</td> </tr>
   <tr> <td>Dual-Path</td><td>VggNet</td> <td>59.4</td><td>86.2</td><td>92.9</td> <td>41.6</td><td>76.3</td><td>87.5</td> </tr>
   <tr> <td>Dual-Path</td><td>ResNet</td> <td>65.6</td><td>89.8</td><td>95.5</td> <td>47.1</td><td>79.9</td><td>90.0</td> </tr>
   <tr> <td>Personality</td><td>ResNeXt, Transformer</td> <td>67.3</td><td>91.7</td><td>96.5</td> <td>--</td><td>--</td><td>--</td> </tr>
   <tr> <td>Align2Ground</td><td>BUTD</td> <td>--</td><td>--</td><td>--</td> <td>56.6</td><td>84.9</td><td>92.8</td> </tr>
   <tr> <td>GXN</td><td>ResNet</td> <td>68.5</td><td>--</td><td>97.9</td> <td>56.6</td><td>--</td><td>94.5</td> </tr>
   <tr> <td>GSLS</td><td>ResNet, BUTD</td> <td>68.9</td><td>94.1</td><td>98.0</td> <td>58.6</td><td>88.2</td><td>94.9</td> </tr>
   <tr> <td>CVSE++</td><td>ResNet</td> <td>69.1</td><td>92.2</td><td>96.1</td> <td>55.6</td><td>86.7</td><td>93.8</td> </tr>
   <tr> <td>PVSE</td><td>ResNet</td> <td>69.2</td><td>91.6</td><td>96.6</td> <td>55.2</td><td>86.5</td><td>93.7</td> </tr>
   <tr> <td>SCO</td><td>VggNet</td> <td>66.6</td><td>91.8</td><td>96.6</td> <td>55.5</td><td>86.6</td><td>93.8</td> </tr>
   <tr> <td>SCO</td><td>ResNet</td> <td>69.9</td><td>92.9</td><td>97.5</td> <td>56.7</td><td>87.5</td><td>94.8</td> </tr>
   <tr> <td>R-SCAN</td><td>BUTD, VrR-VG</td> <td>70.3</td><td>94.5</td><td>98.1</td> <td>57.6</td><td>87.3</td><td>93.7</td> </tr>
   <tr> <td>SAVE</td><td>ResNet</td> <td>70.8</td><td>93.2</td><td>97.6</td> <td>56.9</td><td>87.6</td><td>94.4</td> </tr>
   <tr> <td>MPL</td><td>SCAN_i2t</td> <td>71.1</td><td>93.7</td><td>98.2</td> <td>56.8</td><td>86.7</td><td>93.0</td> </tr>
   <tr> <td>SAEM</td><td>BUTD, Bert</td> <td>71.2</td><td>94.1</td><td>97.7</td> <td>57.8</td><td>88.6</td><td>94.9</td> </tr>
   <tr> <td>OAN</td><td>BUTD</td> <td>71.7</td><td>96.4</td><td>99.3</td> <td>60.2</td><td>88.6</td><td>94.5</td> </tr>
   <tr> <td>GVSE*</td><td>BUTD</td> <td>72.2</td><td>94.1</td><td>98.1</td> <td>60.5</td><td>89.4</td><td>95.8</td> </tr>
   <tr> <td>CAMP</td><td>BUTD</td> <td>72.3</td><td>94.8</td><td>98.3</td> <td>58.5</td><td>87.9</td><td>95.0</td> </tr>
   <tr> <td>CASC</td><td>ResNet</td> <td>72.3</td><td>96.0</td><td>99.0</td> <td>58.9</td><td>89.8</td><td>96.0</td> </tr>
   <tr> <td>SCAN</td><td>BUTD, t2i_AVE</td> <td>70.9</td><td>94.5</td><td>97.8</td> <td>56.4</td><td>87.0</td><td>93.9</td> </tr>
   <tr> <td>SCAN</td><td>BUTD, i2t_AVE</td> <td>69.2</td><td>93.2</td><td>97.5</td> <td>54.4</td><td>86.0</td><td>93.6</td> </tr>
   <tr> <td>SCAN*</td><td>BUTD, LSE+AVE</td> <td>72.7</td><td>94.8</td><td>98.4</td> <td>58.8</td><td>88.4</td><td>94.8</td> </tr>
   <tr> <td>SGM</td><td>BUTD</td> <td>73.4</td><td>93.8</td><td>97.8</td> <td>57.5</td><td>87.3</td><td>94.3</td> </tr>
   <tr> <td>ParNet</td><td>BUTD, NP</td> <td>72.8</td><td>94.9</td><td>97.9</td> <td>57.9</td><td>87.4</td><td>94.0</td> </tr>
   <tr> <td>ParNet</td><td>BUTD, P</td> <td>73.5</td><td>94.5</td><td>98.3</td> <td>58.3</td><td>88.2</td><td>94.1</td> </tr>
   <tr> <td>MTFN</td><td>BUTD</td> <td>71.9</td><td>94.2</td><td>97.9</td> <td>57.3</td><td>88.6</td><td>95.0</td> </tr>
   <tr> <td>MTFN</td><td>BUTD, RR_no_STT</td> <td>74.3</td><td>94.9</td><td>97.9</td> <td>57.5</td><td>88.8</td><td>95.0</td> </tr>
   <tr> <td>MTFN</td><td>BUTD, RR_STT</td> <td>74.3</td><td>94.9</td><td>97.9</td> <td>60.1</td><td>89.1</td><td>95.0</td> </tr>
   <tr> <td>RDAN</td><td>BUTD</td> <td>74.6</td><td>96.2</td><td>98.7</td> <td>61.6</td><td>89.2</td><td>94.7</td> </tr>
   <tr> <td>CVSE^</td><td>BUTD</td> <td>74.8</td><td>95.1</td><td>98.3</td> <td>59.9</td><td>89.4</td><td>95.2</td> </tr>
   <tr> <td>MMCA</td><td>BUTD, Bert</td> <td>74.8</td><td>95.6</td><td>97.7</td> <td>61.6</td><td>89.8</td><td>95.2</td> </tr>
   <tr> <td>BFAN</td><td>BUTD, prob</td> <td>73.0</td><td>94.8</td><td>--</td> <td>58.0</td><td>87.6</td><td>--</td> </tr>
   <tr> <td>BFAN</td><td>BUTD, equal</td> <td>73.7</td><td>94.9</td><td>--</td> <td>58.3</td><td>87.5</td><td>--</td> </tr>
   <tr> <td>BFAN*</td><td>BUTD</td> <td>74.9</td><td>95.2</td><td>--</td> <td>59.4</td><td>88.4</td><td>--</td> </tr>
   <tr> <td>DP-RNN</td><td>BUTD</td> <td>75.3</td><td>95.8</td><td>98.6</td> <td>62.5</td><td>89.7</td><td>95.1</td> </tr>
   <tr> <td>CAAN</td><td>BUTD</td> <td>75.5</td><td>95.4</td><td>98.5</td> <td>61.3</td><td>89.7</td><td>95.2</td> </tr>
   <tr> <td>VSRN*</td><td>BUTD</td> <td>76.2</td><td>94.8</td><td>98.2</td> <td>62.8</td><td>89.7</td><td>95.1</td> </tr>
   <tr> <td>ADAPT</td><td>BUTD, i2t</td> <td>74.5</td><td>94.2</td><td>97.9</td> <td>62.0</td><td>90.4</td><td>95.5</td> </tr>
   <tr> <td>ADAPT</td><td>BUTD, t2i</td> <td>75.3</td><td>95.1</td><td>98.4</td> <td>63.3</td><td>90.0</td><td>95.5</td> </tr>
   <tr> <td>ADAPT*</td><td>BUTD</td> <td>76.5</td><td>95.6</td><td>98.9</td> <td>62.2</td><td>90.5</td><td>96.0</td> </tr>
   <tr> <td>PFAN</td><td>BUTD, t2i</td> <td>75.8</td><td>95.9</td><td>99.0</td> <td>61.0</td><td>89.1</td><td>95.1</td> </tr>
   <tr> <td>PFAN</td><td>BUTD, i2t</td> <td>70.7</td><td>94.1</td><td>97.8</td> <td>53.0</td><td>84.5</td><td>92.6</td> </tr>
   <tr> <td>PFAN*</td><td>BUTD</td> <td>76.5</td><td>96.3</td><td>99.0</td> <td>61.6</td><td>89.6</td><td>95.2</td> </tr>
   <tr> <td>SCG</td><td>VggNet, Prod</td> <td>73.4</td><td>94.8</td><td>97.6</td> <td>56.3</td><td>85.6</td><td>93.5</td> </tr>
   <tr> <td>SCG</td><td>VggNet, Gated</td> <td>76.6</td><td>96.3</td><td>99.2</td> <td>61.4</td><td>88.9</td><td>95.1</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Image</td> <td>76.1</td><td>95.3</td><td>98.2</td> <td>61.0</td><td>88.6</td><td>94.5</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Text</td> <td>74.0</td><td>95.6</td><td>98.4</td> <td>60.6</td><td>88.9</td><td>94.6</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Full</td> <td>76.7</td><td>95.6</td><td>98.5</td> <td>61.7</td><td>89.1</td><td>95.0</td> </tr>
   <tr> <td>ADDR*</td><td>BUTD, SCAN</td> <td>76.1</td><td>95.5</td><td>98.4</td> <td>61.2</td><td>88.9</td><td>94.8</td> </tr>
   <tr> <td>ADDR*</td><td>BUTD, BFAN</td> <td>76.4</td><td>95.8</td><td>98.3</td> <td>62.3</td><td>89.4</td><td>96.2</td> </tr>
   <tr> <td>ADDR*</td><td>BUTD, VSRN</td> <td>77.4</td><td>96.1</td><td>98.9</td> <td>63.5</td><td>90.7</td><td>96.7</td> </tr>
   <tr> <td>AOQ*</td><td>BUTD, SCAN</td> <td>74.1</td><td>95.2</td><td>98.5</td> <td>59.8</td><td>88.6</td><td>95.0</td> </tr>
   <tr> <td>AOQ*</td><td>BUTD, BFAN</td> <td>77.3</td><td>96.0</td><td>98.5</td> <td>61.2</td><td>89.2</td><td>95.0</td> </tr>
   <tr> <td>AOQ*</td><td>BUTD, VSRN</td> <td>77.5</td><td>95.5</td><td>98.6</td> <td>63.5</td><td>90.5</td><td>95.8</td> </tr>
   <tr> <td>HOAD</td><td>BUTD</td> <td>77.0</td><td>96.1</td><td>98.7</td> <td>65.1</td><td>93.1</td><td>97.9</td> </tr>
   <tr> <td>HOAD</td><td>BUTD, +Dist</td> <td>77.8</td><td>96.1</td><td>98.7</td> <td>66.2</td><td>93.0</td><td>97.9</td> </tr>
   <tr> <td>HAL</td><td>SCAN_i2t</td> <td>78.3</td><td>96.3</td><td>98.5</td> <td>60.1</td><td>86.7</td><td>92.8</td> </tr>
   <tr> <td>GSMN</td><td>BUTD, sparse</td> <td>76.1</td><td>95.6</td><td>98.3</td> <td>60.4</td><td>88.7</td><td>95.0</td> </tr>
   <tr> <td>GSMN</td><td>BUTD, dense</td> <td>74.7</td><td>95.3</td><td>98.2</td> <td>60.3</td><td>88.5</td><td>94.6</td> </tr>
   <tr> <td>GSMN*</td><td>BUTD</td> <td>78.4</td><td>96.4</td><td>98.6</td> <td>63.3</td><td>90.1</td><td>95.7</td> </tr>
   <tr> <td>SGRAF</td><td>BUTD, SAF</td> <td>76.1</td><td>95.4</td><td>98.3</td> <td>61.8</td><td>89.4</td><td>95.3</td> </tr>
   <tr> <td>SGRAF</td><td>BUTD, SGR</td> <td>78.0</td><td>95.8</td><td>98.2</td> <td>61.4</td><td>89.3</td><td>95.4</td> </tr>
   <tr> <td>SGRAF*</td><td>BUTD</td> <td>79.6</td><td>96.2</td><td>98.5</td> <td>63.2</td><td>90.7</td><td>96.1</td> </tr>
   <tr> <td>ACMM</td><td>BUTD</td> <td>81.9</td><td>98.0</td><td>99.3</td> <td>58.2</td><td>87.3</td><td>93.9</td> </tr>
   <tr> <td>ACMM*</td><td>BUTD</td> <td>84.1</td><td>97.8</td><td>99.4</td> <td>60.7</td><td>88.7</td><td>94.9</td> </tr>
   <tr> <td>SAN^</td><td>VggNet</td> <td>74.9</td><td>94.9</td><td>98.2</td> <td>60.8</td><td>90.3</td><td>95.7</td> </tr>
   <tr> <td>SAN^</td><td>ResNet</td> <td>85.4</td><td>97.5</td><td>99.0</td> <td>69.1</td><td>93.4</td><td>97.2</td> </tr>
</table> 

### *Performance of MSCOCO5K*
<table>
   <tr> <td rowspan="2">Method_name</td> <td rowspan="2", align="center">Concise_note</td> 
        <td colspan="3", align="center">Sentence retrieval</td> <td colspan="3", align="center">Image retrieval</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td> <td>R@1</td><td>R@5</td><td>R@10</td> </tr>
   <tr> <td>DVSA</td><td>RCNN</td> <td>16.5</td><td>39.2</td><td>52.0</td> <td>10.7</td><td>29.6</td><td>42.2</td> </tr>
   <tr> <td>FV</td><td>GMM+HGLMM</td> <td>17.3</td><td>39.0</td><td>50.2</td> <td>10.8</td><td>28.3</td><td>40.1</td> </tr>
   <tr> <td>Order-emb</td><td>VggNet</td> <td>23.3</td><td>--</td><td>65.0</td> <td>18.0</td><td>--</td><td>57.6</td> </tr>
   <tr> <td>CSE</td><td>ResNet</td> <td>27.9</td><td>57.1</td><td>70.4</td> <td>22.2</td><td>50.2</td><td>64.4</td> </tr>
   <tr> <td>CMPL</td><td>MobileNet</td> <td>24.6</td><td>52.3</td><td>66.4</td> <td>19.1</td><td>44.6</td><td>58.4</td> </tr>
   <tr> <td>CMPM</td><td>ResNet</td> <td>31.1</td><td>60.7</td><td>73.9</td> <td>22.9</td><td>50.2</td><td>63.8</td> </tr>
   <tr> <td>Dual-Path</td><td>VggNet</td> <td>35.5</td><td>63.2</td><td>75.6</td> <td>21.0</td><td>47.5</td><td>60.9</td> </tr>
   <tr> <td>Dual-Path</td><td>ResNet</td> <td>41.2</td><td>70.5</td><td>81.1</td> <td>25.3</td><td>53.4</td><td>66.4</td> </tr>
   <tr> <td>VSE++</td><td>VggNet</td> <td>32.9</td><td>61.7</td><td>74.7</td> <td>24.1</td><td>52.8</td><td>66.2</td> </tr>
   <tr> <td>VSE++</td><td>ResNet</td> <td>41.3</td><td>71.1</td><td>81.2</td> <td>30.3</td><td>59.4</td><td>72.4</td> </tr>
   <tr> <td>GXN</td><td>ResNet</td> <td>42.0</td><td>--</td><td>84.7</td> <td>31.7</td><td>--</td><td>74.6</td> </tr>
   <tr> <td>SCO</td><td>VggNet</td> <td>40.2</td><td>70.1</td><td>81.3</td> <td>31.3</td><td>61.5</td><td>73.9</td> </tr>
   <tr> <td>SCO</td><td>ResNet</td> <td>42.8</td><td>72.3</td><td>83.0</td> <td>33.1</td><td>62.9</td><td>75.5</td> </tr>
   <tr> <td>CVSE++</td><td>ResNet</td> <td>43.2</td><td>73.5</td><td>84.1</td> <td>32.4</td><td>62.2</td><td>74.6</td> </tr>
   <tr> <td>PVSE</td><td>ResNet</td> <td>45.2</td><td>74.3</td><td>84.5</td> <td>32.4</td><td>63.0</td><td>75.0</td> </tr>
   <tr> <td>R-SCAN</td><td>BUTD, VrR-VG</td> <td>45.4</td><td>77.9</td><td>87.9</td> <td>36.2</td><td>65.5</td><td>76.7</td> </tr>
   <tr> <td>SAVE</td><td>ResNet</td> <td>46.7</td><td>76.3</td><td>86.1</td> <td>34.0</td><td>64.8</td><td>77.0</td> </tr>
   <tr> <td>MPL</td><td>SCAN_i2t</td> <td>46.9</td><td>77.7</td><td>87.6</td> <td>34.4</td><td>64.2</td><td>75.9</td> </tr>
   <tr> <td>CASC</td><td>ResNet</td> <td>47.2</td><td>78.3</td><td>87.4</td> <td>34.7</td><td>64.8</td><td>76.8</td> </tr>
   <tr> <td>OAN</td><td>BUTD</td> <td>47.8</td><td>81.2</td><td>90.4</td> <td>37.0</td><td>66.6</td><td>78.0</td> </tr>
   <tr> <td>MTFN</td><td>BUTD</td> <td>44.7</td><td>76.4</td><td>87.3</td> <td>33.1</td><td>64.7</td><td>76.1</td> </tr>
   <tr> <td>MTFN</td><td>BUTD, RR</td> <td>48.3</td><td>77.6</td><td>87.3</td> <td>35.9</td><td>66.1</td><td>76.1</td> </tr>
   <tr> <td>A3VSE</td><td>BUTD</td> <td>49.3</td><td>81.1</td><td>90.2</td> <td>39.0</td><td>68.0</td><td>80.1</td> </tr>
   <tr> <td>GVSE*</td><td>BUTD</td> <td>49.9</td><td>77.4</td><td>87.6</td> <td>38.4</td><td>68.5</td><td>79.7</td> </tr>
   <tr> <td>SGM</td><td>BUTD</td> <td>50.0</td><td>79.3</td><td>87.9</td> <td>35.3</td><td>64.9</td><td>76.5</td> </tr>
   <tr> <td>CAMP</td><td>BUTD</td> <td>50.1</td><td>82.1</td><td>89.7</td> <td>39.0</td><td>68.9</td><td>80.2</td> </tr>   
   <tr> <td>SCAN</td><td>BUTD, i2t_LSE</td> <td>46.4</td><td>77.4</td><td>87.2</td> <td>34.4</td><td>63.7</td><td>75.7</td> </tr>
   <tr> <td>SCAN*</td><td>BUTD, AVE+LSE</td> <td>50.4</td><td>82.2</td><td>90.0</td> <td>38.6</td><td>69.3</td><td>80.4</td> </tr>
   <tr> <td>GOT</td><td>SCAN_i2t</td> <td>50.5</td><td>80.2</td><td>89.8</td> <td>38.1</td><td>66.8</td><td>78.5</td> </tr>
   <tr> <td>HOAD</td><td>BUTD</td> <td>51.2</td><td>81.7</td><td>89.1</td> <td>39.4</td><td>72.5</td><td>84.1</td> </tr>
   <tr> <td>HOAD</td><td>BUTD, +Dist</td> <td>51.4</td><td>81.8</td><td>89.1</td> <td>40.5</td><td>73.5</td><td>84.1</td> </tr>
   <tr> <td>CAAN</td><td>BUTD</td> <td>52.5</td><td>83.3</td><td>90.9</td> <td>41.2</td><td>70.3</td><td>82.9</td> </tr>
   <tr> <td>VSRN*</td><td>BUTD</td> <td>53.0</td><td>81.1</td><td>89.4</td> <td>40.5</td><td>70.6</td><td>81.1</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Image</td> <td>53.2</td><td>82.5</td><td>90.4</td> <td>38.9</td><td>68.5</td><td>79.2</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Text</td> <td>52.0</td><td>81.8</td><td>90.1</td> <td>38.6</td><td>68.1</td><td>79.1</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Full</td> <td>53.7</td><td>83.2</td><td>91.0</td> <td>39.7</td><td>69.1</td><td>79.8</td> </tr>
   <tr> <td>MMCA</td><td>BUTD, Bert</td> <td>54.0</td><td>82.5</td><td>90.7</td> <td>38.7</td><td>69.7</td><td>80.8</td> </tr>
   <tr> <td>SCG</td><td>VggNet, Prod</td> <td>49.9</td><td>78.9</td><td>88.1</td> <td>33.2</td><td>62.4</td><td>74.7</td> </tr>
   <tr> <td>SCG</td><td>VggNet, Gated</td> <td>56.6</td><td>84.5</td><td>92.0</td> <td>39.2</td><td>68.0</td><td>81.3</td> </tr>
   <tr> <td>AOQ*</td><td>BUTD, SCAN</td> <td>51.2</td><td>82.5</td><td>90.1</td> <td>39.4</td><td>69.7</td><td>80.4</td> </tr>
   <tr> <td>AOQ*</td><td>BUTD, VSRN</td> <td>55.1</td><td>83.3</td><td>90.8</td> <td>41.1</td><td>71.5</td><td>82.0</td> </tr>
   <tr> <td>AOQ*</td><td>BUTD, BFAN</td> <td>57.3</td><td>84.5</td><td>91.7</td> <td>40.1</td><td>69.2</td><td>80.1</td> </tr>
   <tr> <td>ADDR*</td><td>BUTD, BFAN</td> <td>54.3</td><td>84.0</td><td>91.5</td> <td>40.1</td><td>69.2</td><td>80.6</td> </tr>
   <tr> <td>ADDR*</td><td>BUTD, VSRN</td> <td>56.6</td><td>85.3</td><td>90.4</td> <td>42.5</td><td>71.9</td><td>82.0</td> </tr>
   <tr> <td>ADDR*</td><td>BUTD, SCAN</td> <td>57.3</td><td>86.0</td><td>92.7</td> <td>41.8</td><td>72.0</td><td>81.3</td> </tr>
   <tr> <td>SGRAF</td><td>BUTD, SAF</td> <td>53.3</td><td>82.3</td><td>90.1</td> <td>39.8</td><td>69.0</td><td>80.2</td> </tr>
   <tr> <td>SGRAF</td><td>BUTD, SGR</td> <td>56.9</td><td>83.2</td><td>90.5</td> <td>40.2</td><td>69.0</td><td>79.8</td> </tr>
   <tr> <td>SGRAF*</td><td>BUTD</td> <td>57.8</td><td>84.9</td><td>91.6</td> <td>41.9</td><td>70.7</td><td>81.3</td> </tr>
   <tr> <td>SAN^</td><td>ResNet</td> <td>65.4</td><td>89.4</td><td>94.8</td> <td>46.2</td><td>77.4</td><td>86.6</td> </tr>
   <tr> <td>ACMM</td><td>BUTD</td> <td>63.5</td><td>88.0</td><td>93.6</td> <td>36.7</td><td>65.1</td><td>76.7</td> </tr>
   <tr> <td>ACMM*</td><td>BUTD</td> <td>66.9</td><td>89.6</td><td>94.9</td> <td>39.5</td><td>69.6</td><td>81.1</td> </tr>
</table> 

### *Performance of CUHK-PEDES*
<table>
   <tr> <td rowspan="2">Method_name</td> <td rowspan="2", align="center">Concise_note</td> 
        <td colspan="3", align="center">Text-to-Image</td> </tr>
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

### *Performance of CUB-Flowers*
<table>
   <tr> <td rowspan="3">Method_name</td> <td rowspan="3", align="center">Concise_note</td> 
        <td colspan="2", align="center">CUB</td> <td colspan="2", align="center">Flowers</td> </tr>
   <tr> <td align="center">Image-to-Text</td> <td align="center">Text-to-Image</td> 
        <td align="center">Image-to-Text</td> <td align="center">Text-to-Image</td> </tr>
   <tr> <td>R@1</td><td>AP@50</td> <td>R@1</td><td>AP@50</td> </tr>
   <tr> <td>FV</td><td>GMM+HGLMM</td> <td>36.5</td><td>35.6</td> <td>54.8</td><td>52.8</td> </tr>
   <tr> <td>Word2Vec</td><td></td> <td>38.6</td><td>33.5</td> <td>54.2</td><td>52.1</td> </tr>
   <tr> <td>Word-NN</td><td>CNN</td> <td>51.0</td><td>43.3</td> <td>60.7</td><td>56.3</td> </tr>
   <tr> <td>Word-NN</td><td>CNN-RNN</td> <td>56.8</td><td>48.7</td> <td>65.6</td><td>59.6</td> </tr>
   <tr> <td>IATV</td><td>Triplet</td> <td>52.5</td><td>52.4</td> <td>64.3</td><td>64.9</td> </tr>
   <tr> <td>IATV</td><td>VggNet</td> <td>61.5</td><td>57.6</td> <td>68.4</td><td>70.1</td> </tr>
   <tr> <td>CMPM</td><td>MobileNet</td> <td>62.1</td><td>64.6</td> <td>66.1</td><td>67.7</td> </tr>
   <tr> <td>CMPL</td><td>MobileNet</td> <td>64.3</td><td>67.9</td> <td>68.9</td><td>69.7</td> </tr>
   <tr> <td>TIMAM</td><td>ResNet, Bert</td> <td>67.7</td><td>70.3</td> <td>70.6</td><td>73.7</td> </tr>
</table> 


## ``Method Summary`` 

### ``*Generic-Feature Extraction*``
**(*NIPS2013_DeViSE*) DeViSE: A Deep Visual-Semantic Embedding Model.** <br>
*Andrea Frome, Greg S. Corrado, Jonathon Shlens, Samy Bengio, Jeffrey Dean, Marc’Aurelio Ranzato, Tomas Mikolov.*<br>
[[paper]](https://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf)

**(*TACL2014_SDT-RNN*) Grounded Compositional Semantics for Finding and Describing Images with Sentences.**<br>
*Richard Socher, Andrej Karpathy, Quoc V. Le, Christopher D. Manning, Andrew Y. Ng.*<br>
[[paper]](https://www.aclweb.org/anthology/Q14-1017.pdf)

**(*NIPSws2014_UVSE*) Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models.**<br>
*Ryan Kiros, Ruslan Salakhutdinov, Richard S. Zemel.*<br>
[[paper]](https://arxiv.org/pdf/1411.2539.pdf)
[[code]](https://github.com/ryankiros/visual-semantic-embedding)
[[demo]](http://www.cs.toronto.edu/~rkiros/lstm_scnlm.html)

**(*NIPS2014_DeFrag*) Deep fragment embeddings for bidirectional image sentence mapping.**<br>
*Andrej Karpathy, Armand Joulin, Li Fei-Fei.*<br>
[[paper]](https://cs.stanford.edu/people/karpathy/nips2014.pdf)

**(*ICCV2015_m-CNN*) Multimodal Convolutional Neural Networks for Matching Image and Sentence.**<br>
*Lin Ma, Zhengdong Lu, Lifeng Shang, Hang Li.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7410658)

**(*CVPR2015_DCCA*) Deep Correlation for Matching Images and Text.**<br>
*Fei Yan, Krystian Mikolajczyk.*<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2015/papers/Yan_Deep_Correlation_for_2015_CVPR_paper.pdf)

**(*CVPR2015_FV*) Associating Neural Word Embeddings with Deep Image Representationsusing Fisher Vectors.**<br>
*Benjamin Klein, Guy Lev, Gil Sadeh, Lior Wolf.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7299073)

**(*CVPR2015_DVSA*) Deep Visual-Semantic Alignments for Generating Image Descriptions.**<br>
*Andrej Karpathy, Li Fei-Fei.*<br>
[[paper]](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)

**(*NIPS2015_STV*) Skip-thought Vectors.**<br>
*Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, Sanja Fidler.*<br>
[[paper]](https://arxiv.org/pdf/1506.06726)

**(*CVPR2016_SPE*) Learning Deep Structure-Preserving Image-Text Embeddings.**<br>
*Liwei Wang, Yin Li, Svetlana Lazebnik.*<br>
[[paper]](http://slazebni.cs.illinois.edu/publications/cvpr16_structure.pdf)

**(*ICCV2017_HM-LSTM*) Hierarchical Multimodal LSTM for Dense Visual-Semantic Embedding.**<br>
*Zhenxing Niu, Mo Zhou, Le Wang, Xinbo Gao, Gang Hua.*<br>
[[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Niu_Hierarchical_Multimodal_LSTM_ICCV_2017_paper.pdf)

**(*ICCV2017_RRF-Net*) Learning a Recurrent Residual Fusion Network for Multimodal Matching.**<br>
*Yu Liu, Yanming Guo, Erwin M. Bakker, Michael S. Lew.*<br>
[[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_a_Recurrent_ICCV_2017_paper.pdf)

**(*CVPR2017_2WayNet*) Linking Image and Text with 2-Way Nets.**<br>
*Aviv Eisenschtat, Lior Wolf.*<br>
[[paper]](https://arxiv.org/pdf/1608.07973)

**(*MM2018_WSJE*) Webly Supervised Joint Embedding for Cross-Modal Image-Text Retrieval.**<br>
*Niluthpol Chowdhury Mithun, Rameswar Panda, Evangelos E. Papalexakis, Amit K. Roy-Chowdhury.*<br>
[[paper]](https://arxiv.org/pdf/1808.07793)

**(*WACV2018_SEAM*) Fast Self-Attentive Multimodal Retrieval.**<br>
*Jônatas Wehrmann, Maurício Armani Lopes, Martin D More, Rodrigo C. Barros.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8354311&tag=1)
[[code]](https://github.com/jwehrmann/seam-retrieval)

**(*CVPR2018_CSE*) End-to-end Convolutional Semantic Embeddings.**<br>
*Quanzeng You, Zhengyou Zhang, Jiebo Luo.*<br>
[[paper]](https://ai.tencent.com/ailab/media/publications/cvpr/End-to-end_Convolutional_Semantic_Embeddings.pdf)

**(*CVPR2018_CHAIN-VSE*) Bidirectional Retrieval Made Simple.**<br>
*Jonatas Wehrmann, Rodrigo C. Barros.*<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wehrmann_Bidirectional_Retrieval_Made_CVPR_2018_paper.pdf)
[[code]](https://github.com/jwehrmann/chain-vse)

**(*CVPR2018_SCO*) Learning Semantic Concepts and Order for Image and Sentence Matching.**<br>
*Yan Huang, Qi Wu, Liang Wang.*<br>
[[paper]](https://arxiv.org/pdf/1712.02036)

**(*MM2019_SAEM*) Learning Fragment Self-Attention Embeddings for Image-Text Matching.**<br>
*Yiling Wu, Shuhui Wang, Guoli Song, Qingming Huang.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3343031.3350940)
[[code]](https://github.com/yiling2018/saem)

**(*ICCV2019_VSRN*) Visual Semantic Reasoning for Image-Text Matching.**<br>
*Kunpeng Li, Yulun Zhang, Kai Li, Yuanyuan Li, Yun Fu.*<br>
[[paper]](https://arxiv.org/pdf/1909.02701.pdf)
[[code]](https://github.com/KunpengLi1994/VSRN)

**(*CVPR2019_Personality*) Engaging Image Captioning via Personality.**<br>
*Kurt Shuster, Samuel Humeau, Hexiang Hu, Antoine Bordes, Jason Weston.*<br>
[[paper]](https://arxiv.org/pdf/1810.10665.pdf)

**(*CVPR2019_PVSE*) Polysemous Visual-Semantic Embedding for Cross-Modal Retrieval.**<br>
*Yale Song, Mohammad Soleymani.*<br>
[[paper]](https://arxiv.org/pdf/1906.04402.pdf)
[[code]](https://github.com/yalesong/pvse)

**(*TOMM2020_NIS*) Upgrading the Newsroom: An Automated Image Selection System for News Articles.**<br>
*Fangyu Liu, Rémi Lebret, Didier Orel, Philippe Sordet, Karl Aberer.*<br>
[[paper]](https://arxiv.org/pdf/2004.11449.pdf)
[[slides]](http://fangyuliu.me/media/others/lsir_talk_final_version_0.3.pdf)
[[demo]](https://modemos.epfl.ch/article)

**(*WACV2020_SGM*) Cross-modal Scene Graph Matching for Relationship-aware Image-Text Retrieval.**<br>
*Sijin Wang, Ruiping Wang, Ziwei Yao, Shiguang Shan, Xilin Chen.*<br>
[[paper]](https://arxiv.org/pdf/1910.05134)

### ``*Cross-Modal Interaction*``
**(*arXiv2014_NIC*) Show and Tell: A Neural Image Caption Generator.**<br>
*Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan.*<br>
[[paper]](https://arxiv.org/pdf/1411.4555)

**(*ICLR2015_m-RNN*) Deep Captioning with Multimodal Recurrent Neural Network(M-RNN).**<br>
*Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Zhiheng Huang, Alan Yuille.*<br>
[[paper]](https://arxiv.org/pdf/1412.6632)
[[code]](https://github.com/mjhucla/mRNN-CR)

**(*CVPR2015_LRCN*) Long-term Recurrent Convolutional Networks for Visual Recognition and Description.**<br>
*Jeff Donahue, Lisa Anne Hendricks, Marcus Rohrbach, Subhashini Venugopalan, Sergio Guadarrama, Kate Saenko, Trevor Darrell.*<br>
[[paper]](https://arxiv.org/pdf/1411.4389)

**(*CVPR2017_DAN*) Dual Attention Networks for Multimodal Reasoning and Matching.**<br>
*Hyeonseob Nam, Jung-Woo Ha, Jeonghee Kim.*<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Nam_Dual_Attention_Networks_CVPR_2017_paper.pdf)

**(*CVPR2017_sm-LSTM*) Instance-aware Image and Sentence Matching with Selective Multimodal LSTM.**<br>
*Yan Huang, Wei Wang, Liang Wang.*<br>
[[paper]](https://arxiv.org/pdf/1611.05588)

**(*ECCV2018_CITE*) Conditional Image-Text Embedding Networks.**<br>
*Bryan A. Plummer, Paige Kordas, M. Hadi Kiapour, Shuai Zheng, Robinson Piramuthu, Svetlana Lazebnik.*<br>
[[paper]](https://arxiv.org/pdf/1711.08389.pdf)

**(*ECCV2018_SCAN*) Stacked Cross Attention for Image-Text Matching.**<br>
*Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He.*<br>
[[paper]](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.pdf)
[[code]](https://github.com/kuanghuei/SCAN)

**(*arXiv2019_R-SCAN*) Learning Visual Relation Priors for Image-Text Matching and Image Captioning with Neural Scene Graph Generators.**<br>
*Kuang-Huei Lee, Hamid Palang, Xi Chen, Houdong Hu, Jianfeng Gao.*<br> 
[[paper]](https://arxiv.org/pdf/1909.09953)

**(*arXiv2019_ParNet*) ParNet: Position-aware Aggregated Relation Network for Image-Text matching.**<br>
*Yaxian Xia, Lun Huang, Wenmin Wang, Xiaoyong Wei, Jie Chen.*<br> 
[[paper]](https://arxiv.org/pdf/1906.06892)

**(*ACML2019_SAVE*) Multi-Scale Visual Semantics Aggregation with Self-Attention for End-to-End Image-Text Matching.**<br>
*Zhuobin Zheng, Youcheng Ben, Chun Yuan.*<br>
[[paper]](http://proceedings.mlr.press/v101/zheng19a/zheng19a.pdf)

**(*ICMR2019_OAN*) Improving What Cross-Modal Retrieval Models Learn through Object-Oriented Inter- and Intra-Modal Attention Networks.**<br>
*Po-Yao Huang, Vaibhav, Xiaojun Chang, Alexander Georg Hauptmann.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3323873.3325043)
[[code]](https://github.com/idejie/OAN)

**(*MM2019_BFAN*) Focus Your Attention: A Bidirectional Focal Attention Network for Image-Text Matching.**<br>
*Chunxiao Liu, Zhendong Mao, An-An Liu, Tianzhu Zhang, Bin Wang, Yongdong Zhang.*<br>
[[paper]](https://arxiv.org/pdf/1909.11416)
[[code]](https://github.com/CrossmodalGroup/BFAN) 

**(*MM2019_MTFN*) Matching Images and Text with Multi-modal Tensor Fusion and Re-ranking.**<br>
*Tan Wang, Xing Xu, Yang Yang, Alan Hanjalic, Heng Tao Shen, Jingkuan Song.*<br>
[[paper]](https://arxiv.org/pdf/1908.04011)
[[code]](https://github.com/Wangt-CN/MTFN-RR-PyTorch-Code) 

**(*IJCAI2019_RDAN*) Multi-Level Visual-Semantic Alignments with Relation-Wise Dual Attention Network for Image and Text Matching.** <br>
*Zhibin Hu, Yongsheng Luo,Jiong Lin,Yan Yan, Jian Chen.*<br>
[[paper]](https://www.ijcai.org/proceedings/2019/0111.pdf)

**(*IJCAI2019_PFAN*) Position Focused Attention Network for Image-Text Matching.**<br>
*Yaxiong Wang, Hao Yang, Xueming Qian, Lin Ma, Jing Lu, Biao Li, Xin Fan.*<br>
[[paper]](https://arxiv.org/pdf/1907.09748)
[[code]](https://github.com/HaoYang0123/Position-Focused-Attention-Network) 

**(*ICCV2019_CAMP*) CAMP: Cross-Modal Adaptive Message Passing for Text-Image Retrieval.**<br>
*Zihao Wang, Xihui Liu, Hongsheng Li, Lu Sheng, Junjie Yan, Xiaogang Wang, Jing Shao.*<br>
[[paper]](https://arxiv.org/pdf/1909.05506.pdf)
[[code]](https://github.com/ZihaoWang-CV/CAMP_iccv19)

**(*ICCV2019_SAN*) Saliency-Guided Attention Network for Image-Sentence Matching.**<br>
*Zhong Ji, Haoran Wang, Jungong Han, Yanwei Pang.*<br>
[[paper]](https://arxiv.org/pdf/1904.09471)
[[code]](https://github.com/HabbakukWang1103/SAN)

**(*TNNLS2020_CASC*) Cross-Modal Attention With Semantic Consistence for Image-Text Matching.**<br>
*Xing Xu, Tan Wang, Yang Yang, Lin Zuo, Fumin Shen, Heng Tao Shen.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8994196)
[[code]](https://github.com/Wangt-CN/Code_CASC) 

**(*AAAI2020_DP-RNN*) Expressing Objects just like Words: Recurrent Visual Embedding for Image-Text Matching.** <br>
*Tianlang Chen, Jiebo Luo.*<br>
[[paper]](https://arxiv.org/pdf/2002.08510.pdf)

**(*AAAI2020_ADAPT*) Adaptive Cross-modal Embeddings for Image-Text Alignment.**<br>
*Jonatas Wehrmann, Camila Kolling, Rodrigo C Barros.*<br>
[[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/6915)
[[code]](https://github.com/jwehrmann/retrieval.pytorch) 

**(*CVPR2020_CAAN*) Context-Aware Attention Network for Image-Text Retrieval.**<br>
*Qi Zhang, Zhen Lei, Zhaoxiang Zhang, Stan Z. Li.*<br>
[[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Context-Aware_Attention_Network_for_Image-Text_Retrieval_CVPR_2020_paper.pdf)

**(*CVPR2020_MMCA*) Multi-Modality Cross Attention Network for Image and Sentence Matching.**<br>
*Xi Wei, Tianzhu Zhang, Yan Li, Yongdong Zhang, Feng Wu.*<br>
[[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Multi-Modality_Cross_Attention_Network_for_Image_and_Sentence_Matching_CVPR_2020_paper.pdf)

**(*CVPR2020_IMRAM*) IMRAM: Iterative Matching with Recurrent Attention Memory for Cross-Modal Image-Text Retrieval.**<br>
*Hui Chen, Guiguang Ding, Xudong Liu, Zijia Lin, Ji Liu, Jungong Han.*<br>
[[paper]](https://arxiv.org/pdf/2003.03772.pdf)
[[code]](https://github.com/HuiChen24/IMRAM)

### ``*Similarity Measurement*``
**(*ICLR2016_Order-emb*) Order-Embeddings of Images and Language.**<br>
*Ivan Vendrov, Ryan Kiros, Sanja Fidler, Raquel Urtasun.*<br>
[[paper]](https://arxiv.org/pdf/1511.06361.pdf)

**(*Access2020_GSLS*) Combining Global and Local Similarity for Cross-Media Retrieval.**<br>
*Zhixin Li, Feng Ling, Canlong Zhang, Huifang Ma.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8970540)

**(*CVPR2020_HOAD*) Visual-Semantic Matching by Exploring High-Order Attention and Distraction.**<br>
*Yongzhi Li, Duo Zhang, Yadong Mu.*<br>
[[paper]](https://pkumyd.github.io/paper/CVPR2020_LYZ.pdf)

**(*CVPR2020_GSMN*) Graph Structured Network for Image-Text Matching.**<br>
*Chunxiao Liu, Zhendong Mao, Tianzhu Zhang, Hongtao Xie, Bin Wang, Yongdong Zhang.*<br>
[[paper]](https://arxiv.org/pdf/2004.00277.pdf)
[[code]](https://github.com/CrossmodalGroup/GSMN)

**(*ICML2020_GOT*) Graph Optimal Transport for Cross-Domain Alignment.**<br>
*Liqun Chen, Zhe Gan, Yu Cheng, Linjie Li, Lawrence Carin, Jingjing Liu.*<br>
[[paper]](https://arxiv.org/pdf/2006.14744.pdf)
[[code]](https://github.com/LiqunChen0606/Graph-Optimal-Transport)

**(*EMNLP2020_WD-Match*) Wasserstein Distance Regularized Sequence Representation for Text Matching in Asymmetrical Domains.**<br>
*Weijie Yu, Chen Xu, Jun Xu, Liang Pang, Xiaopeng Gao, Xiaozhao Wang, Ji-Rong Wen.*<br>
[[paper]](https://arxiv.org/pdf/2010.07717)
[[code]](https://github.com/RUC-WSM/WD-Match)

**(*AAAI2021_SGRAF*) Similarity Reasoning and Filtration for Image-Text Matching.**<br>
*Haiwen Diao, Ying Zhang, Lin Ma, Huchuan Lu.*<br>
[[paper]](https://drive.google.com/file/d/1tAE_qkAxiw1CajjHix9EXoI7xu2t66iQ/view?usp=sharing)
[[code]](https://github.com/Paranioar/SGRAF)

### ``*Loss Function*``
**(*TPAMI2018_TBNN*) Learning Two-Branch Neural Networks for Image-Text Matching Tasks.**<br>
*Liwei Wang, Yin Li, Jing Huang, Svetlana Lazebnik.*<br>
[[paper]](https://arxiv.org/pdf/1704.03470.pdf)
[[code]](https://github.com/lwwang/Two_branch_network)

**(*BMVC2018_VSE++*) VSE++: Improving Visual-Semantic Embeddings with Hard Negatives.**<br>
*Fartash Faghri, David J. Fleet, Jamie Ryan Kiros, Sanja Fidler.*<br>
[[paper]](https://arxiv.org/pdf/1707.05612.pdf)
[[code]](https://github.com/fartashf/vsepp)

**(*ECCV2018_CMPL*) Deep Cross-Modal Projection Learning for Image-Text Matching.**<br>
*Ying Zhang, Huchuan Lu.*<br>
[[paper]](https://drive.google.com/file/d/1aiBuE1NjW83PGgYbP0eQDGEKr4fqMA6J/view)
[[code]](https://github.com/YingZhangDUT/Cross-Modal-Projection-Learning)

**(*ACLws2019_kNN-loss*) A Strong and Robust Baseline for Text-Image Matching.**<br>
*Fangyu Liu, Rongtian Ye.*<br> 
[[paper]](https://www.aclweb.org/anthology/P19-2023.pdf)

**(*ICASSP2019_NAA*) A Neighbor-aware Approach for Image-text Matching.**<br>
*Chunxiao Liu, Zhendong Mao, Wenyu Zang, Bin Wang.*<br> 
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683869)

**(*CVPR2019_PVSE*) Polysemous Visual-Semantic Embedding for Cross-Modal Retrieval.**<br>
*Yale Song, Mohammad Soleymani.*<br>
[[paper]](https://arxiv.org/pdf/1906.04402.pdf)
[[code]](https://github.com/yalesong/pvse)

**(*TOMM2020_Dual-Path*) Dual-path Convolutional Image-Text Embeddings with Instance Loss.**<br>
*Zhedong Zheng, Liang Zheng, Michael Garrett, Yi Yang, Mingliang Xu, YiDong Shen.*<br>
[[paper]](https://arxiv.org/pdf/1711.05535)
[[code]](https://github.com/layumi/Image-Text-Embedding)

**(*AAAI2020_HAL*) HAL: Improved Text-Image Matching by Mitigating Visual Semantic Hubs.**<br>
*Fangyu Liu, Rongtian Ye, Xun Wang, Shuaipeng Li.*<br>
[[paper]](https://arxiv.org/pdf/1911.10097v1.pdf)
[[code]](https://github.com/hardyqr/HAL) 

**(*AAAI2020_CVSE++*) Ladder Loss for Coherent Visual-Semantic Embedding.**<br>
*Mo Zhou, Zhenxing Niu, Le Wang, Zhanning Gao, Qilin Zhang, Gang Hua.*<br>
[[paper]](https://arxiv.org/pdf/1911.07528.pdf)

**(*CVPR2020_MPL*) Universal Weighting Metric Learning for Cross-Modal Matching.**<br>
*Jiwei Wei, Xing Xu, Yang Yang, Yanli Ji, Zheng Wang, Heng Tao Shen.*<br>
[[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Universal_Weighting_Metric_Learning_for_Cross-Modal_Matching_CVPR_2020_paper.pdf)

**(*ECCV2020_PSN*) Preserving Semantic Neighborhoods for Robust Cross-modal Retrieval.**<br>
*Christopher Thomas, Adriana Kovashka.*<br>
[[paper]](https://arxiv.org/pdf/2007.08617.pdf)
[[code]](https://github.com/CLT29/semantic_neighborhoods)

**(*ECCV2020_AOQ*) Adaptive Offline Quintuplet Loss for Image-Text Matching.**<br>
*Tianlang Chen, Jiajun Deng, Jiebo Luo.*<br>
[[paper]](https://arxiv.org/pdf/2003.03669.pdf)
[[code]](https://github.com/sunnychencool/AOQ)

### ``*Un-Supervised or Semi-Supervised*``
**(*ECCV2018_VSA-AE-MMD*) Visual-Semantic Alignment Across Domains Using a Semi-Supervised Approach.**<br>
*Angelo Carraggi, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara.*<br>
[[paper]](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Carraggi_Visual-Semantic_Alignment_Across_Domains_Using_a_Semi-Supervised_Approach_ECCVW_2018_paper.pdf)

**(*MM2019_A3VSE*) Annotation Efficient Cross-Modal Retrieval with Adversarial Attentive Alignment.**<br>
*Po-Yao Huang, Guoliang Kang, Wenhe Liu, Xiaojun Chang, Alexander G Hauptmann.*<br>
[[paper]](http://www.cs.cmu.edu/~poyaoh/data/ann.pdf)

### ``*Zero-Shot or Fewer-Shot*``
**(*CVPR2017_DEM*) Learning a Deep Embedding Model for Zero-Shot Learning.**<br>
*Li Zhang, Tao Xiang, Shaogang Gong.*<br>
[[paper]](https://arxiv.org/pdf/1611.05088.pdf)
[[code]](https://github.com/lzrobots/DeepEmbeddingModel_ZSL)

**(*AAAI2019_GVSE*) Few-shot image and sentence matching via gated visual-semantic matching.**<br>
*Yan Huang, Yang Long, Liang Wang.*<br>
[[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/4866)

**(*ICCV2019_ACMM*) ACMM: Aligned Cross-Modal Memory for Few-Shot Image and Sentence Matching.**<br>
*Yan Huang, Liang Wang.*<br>
[[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_ACMM_Aligned_Cross-Modal_Memory_for_Few-Shot_Image_and_Sentence_Matching_ICCV_2019_paper.pdf)

### ``*Adversarial Learning*``
**(*MM2017_ACMR*) Adversarial Cross-Modal Retrieval.**<br>
*Bokun Wang, Yang Yang, Xing Xu, Alan Hanjalic, Heng Tao Shen.*<br>
[[paper]](http://cfm.uestc.edu.cn/~yangyang/papers/acmr.pdf)
[[code]](https://github.com/sunpeng981712364/ACMR_demo)

**(*COLING2018_CAS*) Learning Visually-Grounded Semantics from Contrastive Adversarial Samples.**<br>
*Haoyue Shi, Jiayuan Mao, Tete Xiao, Yuning Jiang, Jian Sun.*<br>
[[paper]](https://aclweb.org/anthology/C18-1315)
[[code]](https://github.com/ExplorerFreda/VSE-C)

**(*CVPR2018_GXN*) Look, Imagine and Match: Improving Textual-Visual Cross-Modal Retrieval with Generative Models.**<br>
*Jiuxiang Gu, Jianfei Cai, Shafiq Joty, Li Niu, Gang Wang.*<br>
[[paper]](https://arxiv.org/pdf/1711.06420)

**(*ICCV2019_TIMAM*) Adversarial Representation Learning for Text-to-Image Matching.**<br>
*Nikolaos Sarafianos, Xiang Xu, Ioannis A. Kakadiaris.*<br>
[[paper]](https://arxiv.org/pdf/1908.10534.pdf)

**(*CVPR2019_UniVSE*) Unified Visual-Semantic Embeddings: Bridging Vision and Language with Structured Meaning Representations.**<br>
*Hao Wu, Jiayuan Mao, Yufeng Zhang, Yuning Jiang, Lei Li, Weiwei Sun, Wei-Ying Ma.*<br>
[[paper]](https://arxiv.org/pdf/1904.05521.pdf)

**(*arXiv2020_ADDR*) Beyond the Deep Metric Learning: Enhance the Cross-Modal Matching with Adversarial Discriminative Domain Regularization.**<br>
*Li Ren, Kai Li, LiQiang Wang, Kien Hua.*<br>
[[paper]](https://arxiv.org/pdf/2010.12126)

### ``*Commonsense Learning*``
**(*KSEM2019_SCKR*) Semantic Modeling of Textual Relationships in Cross-Modal Retrieval.**<br>
*Jing Yu, Chenghao Yang, Zengchang Qin, Zhuoqian Yang, Yue Hu, Weifeng Zhang.*<br>
[[paper]](https://arxiv.org/pdf/1810.13151)
[[code]](https://github.com/yzhq97/SCKR)

**(*IJCAI2019_SCG*) Knowledge Aware Semantic Concept Expansion for Image-Text Matching.**<br>
*Botian Shi, Lei Ji, Pan Lu, Zhendong Niu, Nan Duan.*<br>
[[paper]](https://www.ijcai.org/Proceedings/2019/0720.pdf)

**(*ECCV2020_CVSE*) Consensus-Aware Visual-Semantic Embedding for Image-Text Matching.**<br>
*Haoran Wang, Ying Zhang, Zhong Ji, Yanwei Pang, Lin Ma.*<br>
[[paper]](https://arxiv.org/pdf/2007.08883.pdf)
[[code]](https://github.com/BruceW91/CVSE)（Corrected codes）

### ``*Identification Learning*``
**(*ICCV2015_LSTM-Q+I*) VQA: Visual question answering.**<br>
*Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, MargaretMitchell, Dhruv Batra, C Lawrence Zitnick, Devi Parikh.*<br>
[[paper]](http://scholar.google.com.hk/scholar_url?url=http://openaccess.thecvf.com/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf&hl=zh-CN&sa=X&ei=EDHkX9aDAY6CywTJ6a2ACw&scisig=AAGBfm2VHgUhZ4sZPI-ODBqcEdCd34_V8w&nossl=1&oi=scholarr)

**(*CVPR2016_Word-NN*) Learning Deep Representations of Fine-grained Visual Descriptions.**<br>
*Scott Reed, Zeynep Akata, Bernt Schiele, Honglak Lee.*<br>
[[paper]](https://arxiv.org/pdf/1605.05395)

**(*CVPR2017_GNA-RNN*) Person search with natural language description.**<br>
*huang  Li, Tong Xiao, Hongsheng Li, Bolei Zhou, DayuYue, Xiaogang Wang.*<br>
[[paper]](https://arxiv.org/pdf/1702.05729)
[[code]](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)

**(*ICCV2017_IATV*) Identity-aware textual-visual matching with latent co-attention.**<br>
*Shuang Li, Tong Xiao, Hongsheng Li, Wei Yang, Xiaogang Wang.*<br>
[[paper]](https://arxiv.org/pdf/1708.01988)

**(*WACV2018_PWM-ATH*) Improving text-based person search by spatial matching and adaptive threshold.**<br>
*Tianlang Chen, Chenliang Xu, Jiebo Luo.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8354312)

**(*ECCV2018_GLA*) Improving deep visual representation for person re-identification by global and local image-language association.**<br>
*Dapeng Chen, Hongsheng Li, Xihui Liu, Yantao Shen, JingShao, Zejian Yuan, Xiaogang Wang.*<br>
[[paper]](https://arxiv.org/pdf/1808.01571)

**(*CVPR2019_DSCMR*) Deep Supervised Cross-modal Retrieval.**<br>
*Liangli Zhen, Peng Hu, Xu Wang, Dezhong Peng.*<br>
[[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhen_Deep_Supervised_Cross-Modal_Retrieval_CVPR_2019_paper.pdf)
[[code]](https://github.com/penghu-cs/DSCMR)

**(*AAAI2020_PMA*) Pose-Guided Multi-Granularity Attention Network for Text-Based Person Search.**<br>
*Ya Jing, Chenyang Si, Junbo Wang, Wei Wang, Liang Wang, Tieniu Tan.*<br>
[[paper]](https://arxiv.org/pdf/1809.08440)

### ``*Scene-Text Learning*``
**(*ECCV2018_SS*) Single Shot Scene Text Retrieval.**<br>
*Lluís Gómez, Andrés Mafla, Marçal Rusiñol, Dimosthenis Karatzas.*<br>
[[paper]](https://arxiv.org/pdf/1808.09044)
[[code_Tensorflow]](https://github.com/lluisgomez/single-shot-str)[[code_Pytorch]](https://github.com/AndresPMD/Pytorch-yolo-phoc)

**(*WACV2020_PHOC*) Fine-grained Image Classification and Retrieval by Combining Visual and Locally Pooled Textual Features.**<br>
*Andres Mafla, Sounak Dey, Ali Furkan Biten, Lluis Gomez, Dimosthenis Karatzas.*<br>
[[paper]](https://arxiv.org/pdf/2001.04732.pdf)
[[code]](https://github.com/AndresPMD/Fine_Grained_Clf)

**(*WACV2021_MMRG*) Multi-Modal Reasoning Graph for Scene-Text Based Fine-Grained Image Classification and Retrieval.**<br>
*Andres Mafla, Sounak Dey, Ali Furkan Biten, Lluis Gomez, Dimosthenis Karatzas.*<br>
[[paper]](https://arxiv.org/pdf/2009.09809)
[[code]](https://github.com/AndresPMD/GCN_classification)

**(*WACV2021_StacMR*) StacMR: Scene-Text Aware Cross-Modal Retrieval.**<br>
*Andrés Mafla, Rafael Sampaio de Rezende, Lluís Gómez, Diane Larlus, Dimosthenis Karatzas.*<br>
[[paper]](https://arxiv.org/pdf/2012.04329)
[[code]](http://europe.naverlabs.com/stacmr)

### ``*Related Works*``
**(*Machine Learning 2010*) Large scale image annotation: learning to rank with joint word-image embeddings.**<br>
*Jason Weston, Samy Bengio, Nicolas Usunier.*<br>
[[paper]](https://link.springer.com/content/pdf/10.1007%2Fs10994-010-5198-3.pdf)

**(*NIPS2013_Word2Vec*) Distributed Representations of Words and Phrases and their Compositionality.**<br>
*Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean.*<br>
[[paper]](https://arxiv.org/pdf/1310.4546)

**(*CVPR2017_DVSQ*) Deep Visual-Semantic Quantization for Efficient Image Retrieval.**<br>
*Yue Cao, Mingsheng Long, Jianmin Wang, Shichen Liu.*<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Cao_Deep_Visual-Semantic_Quantization_CVPR_2017_paper.pdf)

**(*ACL2018_ILU*) Illustrative Language Understanding: Large-Scale Visual Grounding with Image Search.**<br>
*Jamie Kiros, William Chan, Geoffrey Hinton.*<br>
[[paper]](https://aclweb.org/anthology/P18-1085)

**(*AAAI2018_VSE-ens*) VSE-ens: Visual-Semantic Embeddings with Efficient Negative Sampling.**<br>
*Guibing Guo, Songlin Zhai, Fajie Yuan, Yuan Liu, Xingwei Wang.*<br>
[[paper]](https://arxiv.org/pdf/1801.01632.pdf)

**(*ECCV2018_HTG*) An Adversarial Approach to Hard Triplet Generation.**<br>
*Yiru Zhao, Zhongming Jin, Guo-jun Qi, Hongtao Lu, Xian-sheng Hua.*<br>
[[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yiru_Zhao_A_Principled_Approach_ECCV_2018_paper.pdf)

**(*ECCV2018_WebNet*) CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images.**<br>
*Sheng Guo, Weilin Huang, Haozhi Zhang, Chenfan Zhuang, Dengke Dong, Matthew R. Scott, Dinglong Huang.*<br>
[[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sheng_Guo_CurriculumNet_Learning_from_ECCV_2018_paper.pdf)
[[code]](https://github.com/MalongTech/research-curriculumnet)

**(*CVPR2018_BUTD*) Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering.**<br>
*Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, Lei Zhang.*<br>
[[paper]](https://arxiv.org/pdf/1707.07998)
[[code]](https://github.com/peteanderson80/bottom-up-attention)

**(*EMNLP2019_GMMR*) Multi-Head Attention with Diversity for Learning Grounded Multilingual Multimodal Representations.**<br>
*Po-Yao Huang, Xiaojun Chang, Alexander Hauptmann.*<br>
[[paper]](https://www.aclweb.org/anthology/D19-1154.pdf)

**(*EMNLP2019_MIMSD*) Unsupervised Discovery of Multimodal Links in Multi-Image, Multi-Sentence Documents.**<br>
*Jack Hessel, Lillian Lee, David Mimno.*<br>
[[paper]](https://arxiv.org/pdf/1904.07826.pdf)
[[code]](https://github.com/jmhessel/multi-retrieval)

**(*ICCV2019_DRNet*) Fashion Retrieval via Graph Reasoning Networks on a Similarity Pyramid.**<br>
*Zhanghui Kuang, Yiming Gao, Guanbin Li, Ping Luo, Yimin Chen, Liang Lin, Wayne Zhang.*<br>
[[paper]](https://arxiv.org/pdf/1908.11754)

**(*ICCV2019_Align2Ground*) Align2Ground: Weakly Supervised Phrase Grounding Guided by Image-Caption Alignment.**<br>
*Samyak Datta, Karan Sikka, Anirban Roy, Karuna Ahuja, Devi Parikh, Ajay Divakaran.*<br>
[[paper]](https://arxiv.org/pdf/1903.11649.pdf)

**(*CVPR2019_TIRG*) Composing Text and Image for Image Retrieval - An Empirical Odyssey.**<br>
*Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.*<br>
[[paper]](https://arxiv.org/pdf/1812.07119.pdf)

**(*SIGIR2019_PAICM*) Prototype-guided Attribute-wise Interpretable Scheme for Clothing Matching.**<br>
*Xianjing Han, Xuemeng Song, Jianhua Yin, Yinglong Wang, Liqiang Nie.*<br>
[[paper]](https://xuemengsong.github.io/SIGIR2019_PAICM.pdf)

**(*SIGIR2019_NCR*) Neural Compatibility Ranking for Text-based Fashion Matching.**<br>
*Suthee Chaidaroon, Mix Xie, Yi Fang, Alessandro Magnani.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3331184.3331365)

**(*arXiv2020_Tweets*) Deep Multimodal Image-Text Embeddings for Automatic Cross-Media Retrieval.**<br>
*Hadi Abdi Khojasteh, Ebrahim Ansari, Parvin Razzaghi, Akbar Karimi.*<br>
[[paper]](https://arxiv.org/pdf/2002.10016)

**(*arXiv2020_TIMNet*) Weakly-Supervised Feature Learning via Text and Image Matching.**<br>
*Gongbo Liang, Connor Greenwell, Yu Zhang, Xiaoqin Wang, Ramakanth Kavuluru, Nathan Jacobs.*<br>
[[paper]](https://arxiv.org/pdf/2010.03060)
[[code]](http://www.gb-liang.com/TIMNet)

**(*ECCV2020_InfoNCE*) Contrastive Learning for Weakly Supervised Phrase Grounding.**<br>
*Tanmay Gupta, Arash Vahdat, Gal Chechik, Xiaodong Yang, Jan Kautz, Derek Hoiem.*<br>
[[paper]](https://arxiv.org/pdf/2006.09920.pdf)
[[code]](https://github.com/BigRedT/info-ground)

**(*ECCV2020_JVSM*) Learning Joint Visual Semantic Matching Embeddings for Language-guided Retrieval.**<br>
*Yanbei Chen, Loris Bazzani.*<br>
[[paper]](https://assets.amazon.science/5b/db/440af26349adb83c77c85cd11922/learning-joint-visual-semantic-matching-embeddings-for-text-guided-retrieval.pdf)

**(*CVPR2020_POS-SCAN*) More Grounded Image Captioning by Distilling Image-Text Matching Model.**<br>
*Yuanen Zhou, Meng Wang, Daqing Liu, Zhenzhen Hu, Hanwang Zhang.*<br>
[[paper]](https://arxiv.org/pdf/2004.00390)
[[code]](https://github.com/YuanEZhou/Grounded-Image-Captioning)

**(*COLING2020_VSE-Probing*) Probing Multimodal Embeddings for Linguistic Properties: the Visual-Semantic Case.**<br>
*Adam Dahlgren Lindström, Suna Bensch, Johanna Björklund, Frank Drewes.*<br>
[[paper]](https://www.aclweb.org/anthology/2020.coling-main.64.pdf)
[[code]](https://github.com/dali-does/vse-probing)

### ``Posted in``
**(*TC2020_SMAN*) SMAN: Stacked Multimodal Attention Network for Cross-Modal Image-Text Retrieval.**<br>
*Zhong Ji, Haoran Wang, Jungong Han, Yanwei Pang.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9086164)

**(*TCSVT2020_DSRAN*) Learning Dual Semantic Relations with Graph Attention for Image-Text Matching.**<br>
*Keyu Wen, Xiaodong Gu, Qingrong Cheng.*<br>
[[paper]](https://arxiv.org/pdf/2010.11550)
[[code]](https://github.com/kywen1119/DSRAN)

**(*ICPR2020_TERN*) Transformer Reasoning Network for Image-Text Matching and Retrieval.**<br>
*Nicola Messina, Fabrizio Falchi, Andrea Esuli, Giuseppe Amato.*<br>
[[paper]](https://arxiv.org/pdf/2004.09144.pdf)
[[code]](https://github.com/mesnico/TERN)

**(*TOMM2020_TERAN*) Fine-grained Visual Textual Alignment for Cross-Modal Retrieval using Transformer Encoders.**<br>
*Nicola Messina, Giuseppe Amato, Andrea Esuli, Fabrizio Falchi, Claudio Gennaro, Stéphane Marchand-Maillet.*<br>
[[paper]](https://arxiv.org/pdf/2008.05231)
[[code]](https://github.com/mesnico/TERAN)

**(*arXiv2021_PCME*) Probabilistic Embeddings for Cross-Modal Retrieval.**<br>
*Sanghyuk Chun, Seong Joon Oh, Rafael Sampaio de Rezende, Yannis Kalantidis, Diane Larlus.*<br>
[[paper]](https://arxiv.org/pdf/2101.05068)


## ``*Other resources*``
### ``*Fewshot learning*``
[Awesome-Papers-Fewshot](https://github.com/Duan-JM/awesome-papers-fewshot)

### ``*Graph learning*``
[Graph-based Deep Learning Literature](https://github.com/naganandy/graph-based-deep-learning-literature)

### ``*Transformer learning*``
[Recent Advances in Vision and Language PreTrained Models](https://github.com/yuewang-cuhk/awesome-vision-language-pretraining-papers)  

[Vision-Language-Paper-Reading](https://github.com/zh-plus/Vision-Language-Paper-Reading)  

[Awesome Visual-Transformer](https://github.com/dk-liang/Awesome-Visual-Transformer)


## ``*License*``
MIT licensed. See the [LICENSE](LICENSE) file for details. Please contact me at (r1228240468@mail.dlut.edu.cn) if you have any questions.
