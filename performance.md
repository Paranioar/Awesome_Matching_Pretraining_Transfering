Peformance Comparison of Cross-modal Retrieval
==============================


## ``Catalogue ``
* [Commonly-used datasets](#peformance-of-commonly-used-datasets)
    * [Flickr8K](#performance-of-flickr8k)
    * [Flickr30K](#performance-of-flickr30k)
    * [MSCOCO1K](#performance-of-mscoco1k)
    * [MSCOCO5K](#performance-of-mscoco5k)
* [Identity-aware datasets](#peformance-of-identity-aware-datasets)
    * [CUHK-PEDES](#performance-of-cuhk-pedes)
    * [ICFG-PEDES](#performance-of-icfg-pedes)
    * [CUB-Flowers](#performance-of-cub-flowers)


## ``Peformance of Commonly-used Datasets``

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
   <tr> <td>MFM</td><td>VggNet</td> <td>35.6</td><td>67.0</td><td>78.6</td> <td>28.4</td><td>58.5</td><td>72.3</td> </tr>
   <tr> <td>NAA</td><td>ResNet</td> <td>37.2</td><td>68.1</td><td>79.1</td> <td>27.7</td><td>59.6</td><td>71.8</td> </tr>
   <tr> <td>ITMeetsAL</td><td>MobileNet</td> <td>30.9</td><td>58.6</td><td>70.8</td> <td>--</td><td>--</td><td>--</td> </tr>
   <tr> <td>ITMeetsAL</td><td>ResNet</td> <td>40.1</td><td>67.8</td><td>79.2</td> <td>--</td><td>--</td><td>--</td> </tr>
   <tr> <td>2WayNet</td><td>VggNet</td> <td>43.4</td><td>63.2</td><td>--</td> <td>29.3</td><td>49.7</td><td>--</td> </tr>
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
   <tr> <td>MDM</td><td>VggNet</td> <td>44.9</td><td>75.4</td><td>84.4</td> <td>34.4</td><td>67.0</td><td>77.7</td> </tr>
   <tr> <td>RRF-Net</td><td>ResNet</td> <td>47.6</td><td>77.4</td><td>87.1</td> <td>35.4</td><td>68.3</td><td>79.9</td> </tr>
   <tr> <td>CMPL</td><td>MobileNet</td> <td>40.3</td><td>66.9</td><td>76.7</td> <td>30.4</td><td>58.2</td><td>68.5</td> </tr>
   <tr> <td>CMPL</td><td>ResNet</td> <td>49.6</td><td>76.8</td><td>86.1</td> <td>37.3</td><td>65.7</td><td>75.5</td> </tr>
   <tr> <td>2WayNet</td><td>VggNet</td> <td>49.8</td><td>67.5</td><td>--</td> <td>36.0</td><td>55.6</td><td>--</td> </tr>
   <tr> <td>MFM</td><td>VggNet</td> <td>50.2</td><td>78.1</td><td>86.7</td> <td>38.2</td><td>70.1</td><td>80.2</td> </tr>
   <tr> <td>VSE++</td><td>VggNet</td> <td>41.3</td><td>69.1</td><td>77.9</td> <td>31.4</td><td>60.0</td><td>71.2</td> </tr>
   <tr> <td>VSE++</td><td>ResNet</td> <td>52.9</td><td>80.5</td><td>87.2</td> <td>39.6</td><td>70.1</td><td>79.5</td> </tr>
   <tr> <td>TIMAM</td><td>ResNet, Bert</td> <td>53.1</td><td>78.8</td><td>87.6</td> <td>42.6</td><td>71.6</td><td>81.9</td> </tr>
   <tr> <td>TERN</td><td>BUTD, Bert</td> <td>53.2</td><td>79.4</td><td>86.0</td> <td>41.1</td><td>71.9</td><td>81.2</td> </tr>
   <tr> <td>DAN</td><td>VggNet</td> <td>41.4</td><td>73.5</td><td>82.5</td> <td>31.8</td><td>61.7</td><td>72.5</td> </tr>
   <tr> <td>DAN</td><td>ResNet</td> <td>55.0</td><td>81.8</td><td> 89.0</td> <td>39.4</td><td>69.2</td><td>79.1</td> </tr>
   <tr> <td>NAA</td><td>ResNet</td> <td>55.1</td><td>80.3</td><td>89.6</td> <td>39.4</td><td>68.8</td><td>79.9</td> </tr>
   <tr> <td>SCO</td><td>VggNet</td> <td>44.2</td><td>74.1</td><td>83.6</td> <td>32.8</td><td>64.3</td><td>74.9</td> </tr>
   <tr> <td>SCO</td><td>ResNet</td> <td>55.5</td><td>82.0</td><td>89.3</td> <td>41.1</td><td>70.5</td><td>80.1</td> </tr>
   <tr> <td>Dual-Path</td><td>VggNet</td> <td>47.6</td><td>77.3</td><td>87.1</td> <td>35.3</td><td>66.6</td><td>78.2</td> </tr>
   <tr> <td>Dual-Path</td><td>ResNet</td> <td>55.6</td><td>81.9</td><td>89.5</td> <td>39.1</td><td>69.2</td><td>80.9</td> </tr>
   <tr> <td>ITMeetsAL</td><td>VggNet</td> <td>38.5</td><td>66.5</td><td>76.3</td> <td>30.7</td><td>59.4</td><td>70.3</td> </tr>
   <tr> <td>ITMeetsAL</td><td>MobileNet</td> <td>46.6</td><td>73.5</td><td>82.5</td> <td>34.4</td><td>63.3</td><td>74.2</td> </tr>
   <tr> <td>ITMeetsAL</td><td>ResNet</td> <td>56.5</td><td>82.2</td><td>89.6</td> <td>43.5</td><td>71.8</td><td>80.2</td> </tr>
   <tr> <td>CVSE++</td><td>ResNet</td> <td>56.6</td><td>82.5</td><td>90.2</td> <td>42.4</td><td>71.6</td><td>80.8</td> </tr>
   <tr> <td>GXN</td><td>ResNet</td> <td>56.8</td><td>--</td><td>89.6</td> <td>41.5</td><td>--</td><td>80.1</td> </tr>
   <tr> <td>SMAN</td><td>ResNet, Random</td> <td>56.9</td><td>84.8</td><td>91.9</td> <td>43.2</td><td>73.3</td><td>83.5</td> </tr>
   <tr> <td>SMAN</td><td>ResNet, Glove</td> <td>57.3</td><td>85.3</td><td>92.2</td> <td>43.4</td><td>73.7</td><td>83.4</td> </tr>
   <tr> <td>M3A</td><td>ResNet</td> <td>58.1</td><td>82.8</td><td>90.1</td> <td>44.7</td><td>72.4</td><td>81.1</td> </tr>
   <tr> <td>Align2Ground</td><td>BUTD</td> <td>--</td><td>--</td><td>--</td> <td>49.7</td><td>74.8</td><td>83.3</td> </tr>
   <tr> <td>A3VSE</td><td>BUTD</td> <td>65.0</td><td>89.2</td><td>94.5</td> <td>49.5</td><td>79.5</td><td>86.6</td> </tr>
   <tr> <td>DXR</td><td>ResNet, Bert</td> <td>65.1</td><td>87.3</td><td>92.6</td> <td>50.6</td><td>78.8</td><td>86.7</td> </tr>
   <tr> <td>MTFN</td><td>BUTD</td> <td>63.1</td><td>85.8</td><td>92.4</td> <td>46.3</td><td>75.3</td><td>83.6</td> </tr>
   <tr> <td>MTFN</td><td>BUTD, RR_no_STT</td> <td>65.3</td><td>88.3</td><td>93.3</td> <td>46.7</td><td>75.9</td><td>83.8</td> </tr>
   <tr> <td>MTFN</td><td>BUTD, RR_STT</td> <td>65.3</td><td>88.3</td><td>93.3</td> <td>52.0</td><td>80.1</td><td>86.1</td> </tr>
   <tr> <td>R-SCAN</td><td>BUTD, VrR-VG</td> <td>66.3</td><td>90.6</td><td>96.0</td> <td>51.4</td><td>77.8</td><td>84.9</td> </tr>
   <tr> <td>SAVE</td><td>ResNet</td> <td>67.2</td><td>88.3</td><td>94.2</td> <td>49.8</td><td>78.7</td><td>86.2</td> </tr>
   <tr> <td>SCAN</td><td>BUTD, T2I_AVE</td> <td>61.8</td><td>87.5</td><td>93.7</td> <td>45.8</td><td>74.4</td><td>83.0</td> </tr>
   <tr> <td>SCAN</td><td>BUTD, I2T_AVE</td> <td>67.9</td><td>89.0</td><td>94.4</td> <td>43.9</td><td>74.2</td><td>82.8</td> </tr>
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
   <tr> <td>HAL</td><td>SCAN_I2T</td> <td>68.6</td><td>89.9</td><td>94.7</td> <td>46.0</td><td>74.0</td><td>82.3</td> </tr>
   <tr> <td>OAN</td><td>BUTD</td> <td>68.6</td><td>93.0</td><td>96.0</td> <td>53.3</td><td>80.1</td><td>87.1</td> </tr>
   <tr> <td>SAEM</td><td>BUTD, Bert</td> <td>69.1</td><td>91.0</td><td>95.1</td> <td>52.4</td><td>81.1</td><td>88.1</td> </tr>
   <tr> <td>MPL</td><td>SCAN_I2T</td> <td>69.4</td><td>89.9</td><td>95.4</td> <td>47.5</td><td>75.5</td><td>83.1</td> </tr>
   <tr> <td>LIWE</td><td>BUTD, CLMR</td> <td>64.0</td><td>88.3</td><td>93.3</td> <td>46.8</td><td>76.4</td><td>84.5</td> </tr>
   <tr> <td>LIWE</td><td>BUTD, -Glove</td> <td>66.4</td><td>88.9</td><td>94.1</td> <td>47.5</td><td>76.2</td><td>84.9</td> </tr>
   <tr> <td>LIWE</td><td>BUTD, +Glove</td> <td>69.6</td><td>90.3</td><td>95.6</td> <td>51.2</td><td>80.4</td><td>87.2</td> </tr>
   <tr> <td>PFAN</td><td>BUTD, T2I</td> <td>66.0</td><td>89.6</td><td>94.3</td> <td>49.6</td><td>77.0</td><td>84.2</td> </tr>
   <tr> <td>PFAN</td><td>BUTD, I2T</td> <td>67.6</td><td>90.0</td><td>93.8</td> <td>45.7</td><td>74.7</td><td>83.6</td> </tr>
   <tr> <td>PFAN*</td><td>BUTD</td> <td>70.0</td><td>91.8</td><td>95.0</td> <td>50.4</td><td>78.7</td><td>86.1</td> </tr>
   <tr> <td>PFAN++*</td><td>BUTD</td> <td>70.1</td><td>91.8</td><td>96.1</td> <td>52.7</td><td>79.9</td><td>87.0</td> </tr>
   <tr> <td>CAAN</td><td>BUTD</td> <td>70.1</td><td>91.6</td><td>97.2</td> <td>52.8</td><td>79.0</td><td>87.9</td> </tr>
   <tr> <td>DP-RNN</td><td>BUTD</td> <td>70.2</td><td>91.6</td><td>95.8</td> <td>55.5</td><td>81.3</td><td>88.2</td> </tr>
   <tr> <td>TERAN</td><td>BUTD, Bert</td> <td>70.8</td><td>90.9</td><td>95.5</td> <td>56.5</td><td>81.2</td><td>88.2</td> </tr>
   <tr> <td>HOAD</td><td>BUTD</td> <td>70.8</td><td>92.7</td><td>96.0</td> <td>59.5</td><td>85.6</td><td>91.0</td> </tr>
   <tr> <td>HOAD</td><td>BUTD, +Dist</td> <td>70.8</td><td>92.7</td><td>96.0</td> <td>60.9</td><td>86.1</td><td>91.0</td> </tr>
   <tr> <td>GOT</td><td>SCAN_I2T</td> <td>70.9</td><td>92.8</td><td>95.5</td> <td>50.7</td><td>78.7</td><td>86.2</td> </tr>
   <tr> <td>LGSGM</td><td>BUTD</td> <td>71.0</td><td>91.9</td><td>96.1</td> <td>57.4</td><td>84.1</td><td>90.2</td> </tr>
   <tr> <td>VSRN</td><td>BUTD</td> <td>70.4</td><td>89.2</td><td>93.7</td> <td>53.0</td><td>77.9</td><td>85.7</td> </tr>
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
   <tr> <td>CVSE</td><td>BUTD</td> <td>73.5</td><td>92.1</td><td>95.8</td> <td>52.9</td><td>80.4</td><td>87.8</td> </tr>
   <tr> <td>SMFEA</td><td>BUTD</td> <td>73.7</td><td>92.5</td><td>96.1</td> <td>54.7</td><td>82.1</td><td>88.4</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Image</td> <td>67.0</td><td>90.5</td><td>95.6</td> <td>51.2</td><td>78.2</td><td>85.5</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Text</td> <td>68.8</td><td>91.6</td><td>96.0</td> <td>53.0</td><td>79.0</td><td>87.1</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Full</td> <td>74.1</td><td>93.0</td><td>96.6</td> <td>53.9</td><td>79.4</td><td>87.2</td> </tr>
   <tr> <td>MMCA</td><td>BUTD, Bert</td> <td>74.2</td><td>92.8</td><td>96.4</td> <td>54.8</td><td>81.4</td><td>87.8</td> </tr>
   <tr> <td>SHAN</td><td>BUTD, T2I</td> <td>72.5</td><td>92.3</td><td>95.8</td> <td>53.6</td><td>78.6</td><td>85.5</td> </tr>
   <tr> <td>SHAN</td><td>BUTD, I2T</td> <td>70.6</td><td>91.7</td><td>95.5</td> <td>50.5</td><td>77.1</td><td>85.2</td> </tr>
   <tr> <td>SHAN</td><td>BUTD, Full</td> <td>74.6</td><td>93.5</td><td>96.9</td> <td>55.3</td><td>81.3</td><td>88.4</td> </tr>
   <tr> <td>CCRS*</td><td>BUTD, SCAN</td> <td>70.1</td><td>92.0</td><td>96.0</td> <td>52.3</td><td>79.9</td><td>86.8</td> </tr>
   <tr> <td>CCRS*</td><td>BUTD, BFAN</td> <td>75.3</td><td>93.6</td><td>96.7</td> <td>55.4</td><td>81.3</td><td>87.7</td> </tr>
   <tr> <td>SAN^</td><td>VggNet</td> <td>67.0</td><td>88.0</td><td>94.6</td> <td>51.4</td><td>77.2</td><td>85.2</td> </tr>
   <tr> <td>SAN^</td><td>ResNet</td> <td>75.5</td><td>92.6</td><td>96.2</td> <td>60.1</td><td>84.7</td><td>90.6</td> </tr>
   <tr> <td>GSMN</td><td>BUTD, sparse</td> <td>71.4</td><td>92.0</td><td>96.1</td> <td>53.9</td><td>79.7</td><td>87.1</td> </tr>
   <tr> <td>GSMN</td><td>BUTD, dense</td> <td>72.6</td><td>93.5</td><td>96.8</td> <td>53.7</td><td>80.0</td><td>87.0</td> </tr>
   <tr> <td>GSMN*</td><td>BUTD</td> <td>76.4</td><td>94.3</td><td>97.3</td> <td>57.4</td><td>82.3</td><td>89.0</td> </tr>
   <tr> <td>ADAPT</td><td>BUTD, I2T</td> <td>70.2</td><td>90.8</td><td>95.8</td> <td>55.5</td><td>82.7</td><td>89.8</td> </tr>
   <tr> <td>ADAPT</td><td>BUTD, T2I</td> <td>73.6</td><td>93.7</td><td>96.7</td> <td>57.0</td><td>83.6</td><td>90.3</td> </tr>
   <tr> <td>ADAPT*</td><td>BUTD, +GloVe</td> <td>76.6</td><td>95.4</td><td>97.6</td> <td>60.7</td><td>86.6</td><td>92.0</td> </tr>
   <tr> <td>SGRAF</td><td>BUTD, SAF</td> <td>73.7</td><td>93.3</td><td>96.3</td> <td>56.1</td><td>81.5</td><td>88.0</td> </tr>
   <tr> <td>SGRAF</td><td>BUTD, SGR</td> <td>75.2</td><td>93.3</td><td>96.6</td> <td>56.2</td><td>81.0</td><td>86.5</td> </tr>
   <tr> <td>SGRAF*</td><td>BUTD</td> <td>77.8</td><td>94.1</td><td>97.4</td> <td>58.5</td><td>83.0</td><td>88.8</td> </tr>
   <tr> <td>DSRAN</td><td>BUTD, GRU</td> <td>72.6</td><td>93.6</td><td>96.3</td> <td>56.3</td><td>84.0</td><td>89.8</td> </tr>
   <tr> <td>DSRAN</td><td>BUTD, Bert</td> <td>75.3</td><td>94.4</td><td>97.6</td> <td>57.3</td><td>84.8</td><td>90.9</td> </tr>
   <tr> <td>DSRAN*</td><td>BUTD, GRU</td> <td>74.9</td><td>94.5</td><td>97.0</td> <td>58.6</td><td>85.8</td><td>91.3</td> </tr>
   <tr> <td>DSRAN*</td><td>BUTD, Bert</td> <td>77.8</td><td>95.1</td><td>97.6</td> <td>59.2</td><td>86.0</td><td>91.9</td> </tr>
   <tr> <td>CAMERA</td><td>BUTD, Bert</td> <td>76.5</td><td>95.1</td><td>97.2</td> <td>58.9</td><td>84.7</td><td>90.2</td> </tr>
   <tr> <td>CAMERA*</td><td>BUTD, Bert</td> <td>78.0</td><td>95.1</td><td>97.9</td> <td>60.3</td><td>85.9</td><td>91.7</td> </tr>
   <tr> <td>T-EMDE</td><td>BUTD, SAF</td> <td>75.2</td><td>94.2</td><td>97.1</td> <td>57.1</td><td>82.2</td><td>88.3</td> </tr>
   <tr> <td>T-EMDE</td><td>BUTD, SGR</td> <td>77.5</td><td>93.1</td><td>97.2</td> <td>56.9</td><td>82.0</td><td>87.5</td> </tr>
   <tr> <td>T-EMDE*</td><td>BUTD, SGRAF</td> <td>78.8</td><td>94.4</td><td>97.5</td> <td>59.6</td><td>83.6</td><td>89.2</td> </tr>
   <tr> <td>DIME</td><td>BUTD, I2T, Bert</td> <td>77.4</td><td>95.0</td><td>97.4</td> <td>60.1</td><td>85.5</td><td>91.8</td> </tr>
   <tr> <td>DIME</td><td>BUTD, T2I, Bert</td> <td>77.5</td><td>93.5</td><td>97.5</td> <td>59.1</td><td>85.5</td><td>91.0</td> </tr>
   <tr> <td>DIME*</td><td>BUTD, Bert</td> <td>81.0</td><td>95.9</td><td>98.4</td> <td>63.6</td><td>88.1</td><td>93.0</td> </tr>
   <tr> <td>PG*</td><td>BUTD, +3loss</td> <td>81.0</td><td>94.5</td><td>97.1</td> <td>60.6</td><td>86.5</td><td>92.4</td> </tr>
   <tr> <td>PG*</td><td>BUTD, +GloVe</td> <td>82.8</td><td>95.9</td><td>97.9</td> <td>62.2</td><td>89.3</td><td>93.8</td> </tr>
   <tr> <td>ACMM</td><td>BUTD</td> <td>80.0</td><td>95.5</td><td>98.2</td> <td>50.2</td><td>76.8</td><td>84.7</td> </tr>
   <tr> <td>ACMM*</td><td>BUTD</td> <td>85.2</td><td>96.7</td><td>98.4</td> <td>53.8</td><td>79.8</td><td>86.8</td> </tr>
   <tr> <td>GPO</td><td>IN, BiGRU</td> <td>77.1</td><td>94.5</td><td>97.1</td> <td>58.5</td><td>84.1</td><td>89.6</td> </tr>
   <tr> <td>GPO*</td><td>IN+VG, BiGRU</td> <td>80.7</td><td>96.4</td><td>98.3</td> <td>60.8</td><td>86.3</td><td>92.3</td> </tr>
   <tr> <td>GPO*</td><td>IN+VG, Bert</td> <td>85.3</td><td>97.2</td><td>98.9</td> <td>66.7</td><td>89.9</td><td>94.0</td> </tr>
   <tr> <td>GPO*</td><td>WSL, Bert</td> <td>88.7</td><td>98.9</td><td>99.8</td> <td>76.1</td><td>94.5</td><td>97.1</td> </tr>
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
   <tr> <td>MDM</td><td>VggNet</td> <td>54.7</td><td>84.1</td><td>91.9</td> <td>44.6</td><td>79.6</td><td>90.5</td> </tr>
   <tr> <td>2WayNet</td><td>VggNet</td> <td>55.8</td><td>75.2</td><td>--</td> <td>39.7</td><td>63.3</td><td>--</td> </tr>
   <tr> <td>CMPM</td><td>ResNet</td> <td>56.1</td><td>86.3</td><td>92.9</td> <td>44.6</td><td>78.8</td><td>89.0</td> </tr>
   <tr> <td>CSE</td><td>ResNet</td> <td>56.3</td><td>84.4</td><td>92.2</td> <td>45.7</td><td>81.2</td><td>90.6</td> </tr>    
   <tr> <td>RRF-Net</td><td>ResNet</td> <td>56.4</td><td>85.3</td><td>91.5</td> <td>43.9</td><td>78.1</td><td>88.6</td> </tr>
   <tr> <td>ITMeetsAL</td><td>VggNet</td> <td>44.2</td><td>76.1</td><td>86.3</td> <td>37.1</td><td>72.7</td><td>85.1</td> </tr>
   <tr> <td>ITMeetsAL</td><td>MobileNet</td> <td>54.7</td><td>84.3</td><td>91.1</td> <td>41.0</td><td>76.7</td><td>88.1</td> </tr>
   <tr> <td>ITMeetsAL</td><td>ResNet</td> <td>58.5</td><td>85.3</td><td>92.1</td> <td>48.3</td><td>82.0</td><td>90.6</td> </tr>
   <tr> <td>MFM</td><td>VggNet</td> <td>58.9</td><td>86.3</td><td>92.4</td> <td>47.7</td><td>81.0</td><td>90.9</td> </tr>
   <tr> <td>CHAIN-VSE</td><td>VggNet</td> <td>51.6</td><td>82.0</td><td>91.3</td> <td>38.6</td><td>75.1</td><td>87.2</td> </tr>
   <tr> <td>CHAIN-VSE</td><td>ResNet</td> <td>59.4</td><td>88.0</td><td>94.2</td> <td>43.5</td><td>79.8</td><td>90.2</td> </tr>
   <tr> <td>NAA</td><td>ResNet</td> <td>61.3</td><td>87.9</td><td>95.4</td> <td>47.0</td><td>80.8</td><td>90.1</td> </tr>
   <tr> <td>TERN</td><td>BUTD, Bert</td> <td>63.7</td><td>90.5</td><td>96.2</td> <td>51.9</td><td>85.6</td><td>93.6</td> </tr>
   <tr> <td>VSE++</td><td>VggNet</td> <td>57.2</td><td>86.0</td><td>93.3</td> <td>45.9</td><td>79.4</td><td>89.1</td> </tr>
   <tr> <td>VSE++</td><td>ResNet</td> <td>64.6</td><td>90.0</td><td>95.7</td> <td>52.0</td><td>84.3</td><td>92.0</td> </tr>
   <tr> <td>Dual-Path</td><td>VggNet</td> <td>59.4</td><td>86.2</td><td>92.9</td> <td>41.6</td><td>76.3</td><td>87.5</td> </tr>
   <tr> <td>Dual-Path</td><td>ResNet</td> <td>65.6</td><td>89.8</td><td>95.5</td> <td>47.1</td><td>79.9</td><td>90.0</td> </tr>
   <tr> <td>DXR</td><td>ResNet, Bert</td> <td>67.0</td><td>93.0</td><td>97.6</td> <td>56.8</td><td>88.2</td><td>94.9</td> </tr>
   <tr> <td>Personality</td><td>ResNeXt, Transformer</td> <td>67.3</td><td>91.7</td><td>96.5</td> <td>--</td><td>--</td><td>--</td> </tr>
   <tr> <td>Align2Ground</td><td>BUTD</td> <td>--</td><td>--</td><td>--</td> <td>56.6</td><td>84.9</td><td>92.8</td> </tr>
   <tr> <td>SMAN</td><td>ResNet, Random</td> <td>67.9</td><td>90.6</td><td>96.2</td> <td>58.8</td><td>87.0</td><td>93.7</td> </tr>
   <tr> <td>SMAN</td><td>ResNet, Glove</td> <td>68.4</td><td>91.3</td><td>96.6</td> <td>58.5</td><td>87.4</td><td>93.5</td> </tr>
   <tr> <td>GXN</td><td>ResNet</td> <td>68.5</td><td>--</td><td>97.9</td> <td>56.6</td><td>--</td><td>94.5</td> </tr>
   <tr> <td>GSLS</td><td>ResNet, BUTD</td> <td>68.9</td><td>94.1</td><td>98.0</td> <td>58.6</td><td>88.2</td><td>94.9</td> </tr>
   <tr> <td>CVSE++</td><td>ResNet</td> <td>69.1</td><td>92.2</td><td>96.1</td> <td>55.6</td><td>86.7</td><td>93.8</td> </tr>
   <tr> <td>PVSE</td><td>ResNet</td> <td>69.2</td><td>91.6</td><td>96.6</td> <td>55.2</td><td>86.5</td><td>93.7</td> </tr>
   <tr> <td>DSVE-Loc</td><td>ResNet</td> <td>69.8</td><td>91.9</td><td>96.6</td> <td>55.9</td><td>86.9</td><td>94.0</td> </tr>
   <tr> <td>SCO</td><td>VggNet</td> <td>66.6</td><td>91.8</td><td>96.6</td> <td>55.5</td><td>86.6</td><td>93.8</td> </tr>
   <tr> <td>SCO</td><td>ResNet</td> <td>69.9</td><td>92.9</td><td>97.5</td> <td>56.7</td><td>87.5</td><td>94.8</td> </tr>
   <tr> <td>R-SCAN</td><td>BUTD, VrR-VG</td> <td>70.3</td><td>94.5</td><td>98.1</td> <td>57.6</td><td>87.3</td><td>93.7</td> </tr>
   <tr> <td>M3A</td><td>ResNet</td> <td>70.4</td><td>91.7</td><td>96.8</td> <td>58.4</td><td>87.1</td><td>94.0</td> </tr>
   <tr> <td>SAVE</td><td>ResNet</td> <td>70.8</td><td>93.2</td><td>97.6</td> <td>56.9</td><td>87.6</td><td>94.4</td> </tr>
   <tr> <td>MPL</td><td>SCAN_I2T</td> <td>71.1</td><td>93.7</td><td>98.2</td> <td>56.8</td><td>86.7</td><td>93.0</td> </tr>
   <tr> <td>SAEM</td><td>BUTD, Bert</td> <td>71.2</td><td>94.1</td><td>97.7</td> <td>57.8</td><td>88.6</td><td>94.9</td> </tr>
   <tr> <td>SoDeep</td><td>DSVE-Loc</td> <td>71.5</td><td>92.8</td><td>97.1</td> <td>56.2</td><td>87.0</td><td>94.3</td> </tr>
   <tr> <td>OAN</td><td>BUTD</td> <td>71.7</td><td>96.4</td><td>99.3</td> <td>60.2</td><td>88.6</td><td>94.5</td> </tr>
   <tr> <td>GVSE*</td><td>BUTD</td> <td>72.2</td><td>94.1</td><td>98.1</td> <td>60.5</td><td>89.4</td><td>95.8</td> </tr>
   <tr> <td>CAMP</td><td>BUTD</td> <td>72.3</td><td>94.8</td><td>98.3</td> <td>58.5</td><td>87.9</td><td>95.0</td> </tr>
   <tr> <td>CASC</td><td>ResNet</td> <td>72.3</td><td>96.0</td><td>99.0</td> <td>58.9</td><td>89.8</td><td>96.0</td> </tr>
   <tr> <td>SCAN</td><td>BUTD, T2I_AVE</td> <td>70.9</td><td>94.5</td><td>97.8</td> <td>56.4</td><td>87.0</td><td>93.9</td> </tr>
   <tr> <td>SCAN</td><td>BUTD, I2T_AVE</td> <td>69.2</td><td>93.2</td><td>97.5</td> <td>54.4</td><td>86.0</td><td>93.6</td> </tr>
   <tr> <td>SCAN*</td><td>BUTD, LSE+AVE</td> <td>72.7</td><td>94.8</td><td>98.4</td> <td>58.8</td><td>88.4</td><td>94.8</td> </tr>
   <tr> <td>LIWE</td><td>BUTD, -Glove</td> <td>69.6</td><td>93.9</td><td>98.0</td> <td>55.5</td><td>87.3</td><td>94.2</td> </tr>
   <tr> <td>LIWE</td><td>BUTD, CLMR</td> <td>71.8</td><td>93.1</td><td>97.6</td> <td>56.2</td><td>87.5</td><td>94.2</td> </tr>
   <tr> <td>LIWE</td><td>BUTD, +Glove</td> <td>73.2</td><td>95.5</td><td>98.2</td> <td>57.9</td><td>88.3</td><td>94.5</td> </tr>
   <tr> <td>SGM</td><td>BUTD</td> <td>73.4</td><td>93.8</td><td>97.8</td> <td>57.5</td><td>87.3</td><td>94.3</td> </tr>
   <tr> <td>ParNet</td><td>BUTD, NP</td> <td>72.8</td><td>94.9</td><td>97.9</td> <td>57.9</td><td>87.4</td><td>94.0</td> </tr>
   <tr> <td>ParNet</td><td>BUTD, P</td> <td>73.5</td><td>94.5</td><td>98.3</td> <td>58.3</td><td>88.2</td><td>94.1</td> </tr>
   <tr> <td>MTFN</td><td>BUTD</td> <td>71.9</td><td>94.2</td><td>97.9</td> <td>57.3</td><td>88.6</td><td>95.0</td> </tr>
   <tr> <td>MTFN</td><td>BUTD, RR_no_STT</td> <td>74.3</td><td>94.9</td><td>97.9</td> <td>57.5</td><td>88.8</td><td>95.0</td> </tr>
   <tr> <td>MTFN</td><td>BUTD, RR_STT</td> <td>74.3</td><td>94.9</td><td>97.9</td> <td>60.1</td><td>89.1</td><td>95.0</td> </tr>
   <tr> <td>RDAN</td><td>BUTD</td> <td>74.6</td><td>96.2</td><td>98.7</td> <td>61.6</td><td>89.2</td><td>94.7</td> </tr>
   <tr> <td>CVSE</td><td>BUTD</td> <td>74.8</td><td>95.1</td><td>98.3</td> <td>59.9</td><td>89.4</td><td>95.2</td> </tr>
   <tr> <td>MMCA</td><td>BUTD, Bert</td> <td>74.8</td><td>95.6</td><td>97.7</td> <td>61.6</td><td>89.8</td><td>95.2</td> </tr>
   <tr> <td>BFAN</td><td>BUTD, prob</td> <td>73.0</td><td>94.8</td><td>--</td> <td>58.0</td><td>87.6</td><td>--</td> </tr>
   <tr> <td>BFAN</td><td>BUTD, equal</td> <td>73.7</td><td>94.9</td><td>--</td> <td>58.3</td><td>87.5</td><td>--</td> </tr>
   <tr> <td>BFAN*</td><td>BUTD</td> <td>74.9</td><td>95.2</td><td>--</td> <td>59.4</td><td>88.4</td><td>--</td> </tr>
   <tr> <td>SMFEA</td><td>BUTD</td> <td>75.1</td><td>95.4</td><td>98.3</td> <td>62.5</td><td>90.1</td><td>96.2</td> </tr>
   <tr> <td>DP-RNN</td><td>BUTD</td> <td>75.3</td><td>95.8</td><td>98.6</td> <td>62.5</td><td>89.7</td><td>95.1</td> </tr>
   <tr> <td>CCRS*</td><td>BUTD, SCAN</td> <td>70.9</td><td>94.3</td><td>98.0</td> <td>57.3</td><td>87.6</td><td>94.3</td> </tr>
   <tr> <td>CCRS*</td><td>BUTD, BFAN</td> <td>75.4</td><td>95.3</td><td>98.5</td> <td>60.3</td><td>88.6</td><td>94.6</td> </tr>
   <tr> <td>CAAN</td><td>BUTD</td> <td>75.5</td><td>95.4</td><td>98.5</td> <td>61.3</td><td>89.7</td><td>95.2</td> </tr>
   <tr> <td>VSRN</td><td>BUTD</td> <td>74.0</td><td>94.3</td><td>97.8</td> <td>60.8</td><td>88.4</td><td>94.1</td> </tr>
   <tr> <td>VSRN*</td><td>BUTD</td> <td>76.2</td><td>94.8</td><td>98.2</td> <td>62.8</td><td>89.7</td><td>95.1</td> </tr>
   <tr> <td>ADAPT</td><td>BUTD, I2T</td> <td>74.5</td><td>94.2</td><td>97.9</td> <td>62.0</td><td>90.4</td><td>95.5</td> </tr>
   <tr> <td>ADAPT</td><td>BUTD, T2I</td> <td>75.3</td><td>95.1</td><td>98.4</td> <td>63.3</td><td>90.0</td><td>95.5</td> </tr>
   <tr> <td>ADAPT*</td><td>BUTD</td> <td>76.5</td><td>95.6</td><td>98.9</td> <td>62.2</td><td>90.5</td><td>96.0</td> </tr>
   <tr> <td>PFAN</td><td>BUTD, T2I</td> <td>75.8</td><td>95.9</td><td>99.0</td> <td>61.0</td><td>89.1</td><td>95.1</td> </tr>
   <tr> <td>PFAN</td><td>BUTD, I2T</td> <td>70.7</td><td>94.1</td><td>97.8</td> <td>53.0</td><td>84.5</td><td>92.6</td> </tr>
   <tr> <td>PFAN*</td><td>BUTD</td> <td>76.5</td><td>96.3</td><td>99.0</td> <td>61.6</td><td>89.6</td><td>95.2</td> </tr>
   <tr> <td>SCG</td><td>VggNet, Prod</td> <td>73.4</td><td>94.8</td><td>97.6</td> <td>56.3</td><td>85.6</td><td>93.5</td> </tr>
   <tr> <td>SCG</td><td>VggNet, Gated</td> <td>76.6</td><td>96.3</td><td>99.2</td> <td>61.4</td><td>88.9</td><td>95.1</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Image</td> <td>76.1</td><td>95.3</td><td>98.2</td> <td>61.0</td><td>88.6</td><td>94.5</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Text</td> <td>74.0</td><td>95.6</td><td>98.4</td> <td>60.6</td><td>88.9</td><td>94.6</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Full</td> <td>76.7</td><td>95.6</td><td>98.5</td> <td>61.7</td><td>89.1</td><td>95.0</td> </tr>
   <tr> <td>SHAN</td><td>BUTD, T2I</td> <td>75.9</td><td>96.1</td><td>98.7</td> <td>60.7</td><td>88.2</td><td>94.2</td> </tr>
   <tr> <td>SHAN</td><td>BUTD, I2T</td> <td>73.0</td><td>95.8</td><td>97.9</td> <td>58.5</td><td>87.3</td><td>94.0</td> </tr>
   <tr> <td>SHAN</td><td>BUTD, Full</td> <td>76.8</td><td>96.3</td><td>98.7</td> <td>62.6</td><td>89.6</td><td>95.8</td> </tr>
   <tr> <td>PFAN++*</td><td>BUTD</td> <td>77.1</td><td>96.5</td><td>98.3</td> <td>62.5</td><td>89.9</td><td>95.4</td> </tr>
   <tr> <td>ADDR*</td><td>BUTD, SCAN</td> <td>76.1</td><td>95.5</td><td>98.4</td> <td>61.2</td><td>88.9</td><td>94.8</td> </tr>
   <tr> <td>ADDR*</td><td>BUTD, BFAN</td> <td>76.4</td><td>95.8</td><td>98.3</td> <td>62.3</td><td>89.4</td><td>96.2</td> </tr>
   <tr> <td>ADDR*</td><td>BUTD, VSRN</td> <td>77.4</td><td>96.1</td><td>98.9</td> <td>63.5</td><td>90.7</td><td>96.7</td> </tr>
   <tr> <td>CAMERA</td><td>BUTD, Bert</td> <td>75.9</td><td>95.5</td><td>98.6</td> <td>62.3</td><td>90.1</td><td>95.2</td> </tr>
   <tr> <td>CAMERA*</td><td>BUTD, Bert</td> <td>77.5</td><td>96.3</td><td>98.8</td> <td>63.4</td><td>90.9</td><td>95.8</td> </tr>
   <tr> <td>AOQ*</td><td>BUTD, SCAN</td> <td>74.1</td><td>95.2</td><td>98.5</td> <td>59.8</td><td>88.6</td><td>95.0</td> </tr>
   <tr> <td>AOQ*</td><td>BUTD, BFAN</td> <td>77.3</td><td>96.0</td><td>98.5</td> <td>61.2</td><td>89.2</td><td>95.0</td> </tr>
   <tr> <td>AOQ*</td><td>BUTD, VSRN</td> <td>77.5</td><td>95.5</td><td>98.6</td> <td>63.5</td><td>90.5</td><td>95.8</td> </tr>
   <tr> <td>TERAN</td><td>BUTD, Bert</td> <td>77.7</td><td>95.9</td><td>98.6</td> <td>65.0</td><td>91.2</td><td>96.4</td> </tr>
   <tr> <td>HOAD^</td><td>BUTD</td> <td>77.0</td><td>96.1</td><td>98.7</td> <td>65.1</td><td>93.1</td><td>97.9</td> </tr>
   <tr> <td>HOAD^</td><td>BUTD, +Dist</td> <td>77.8</td><td>96.1</td><td>98.7</td> <td>66.2</td><td>93.0</td><td>97.9</td> </tr>
   <tr> <td>TOD-Net</td><td>VSE++</td> <td>68.6</td><td>92.0</td><td>96.9</td> <td>54.5</td><td>85.3</td><td>92.4</td> </tr>
   <tr> <td>TOD-Net</td><td>Bert</td> <td>75.8</td><td>95.3</td><td>98.4</td> <td>61.8</td><td>89.6</td><td>95.0</td> </tr>
   <tr> <td>TOD-Net*</td><td>Bert</td> <td>78.1</td><td>96.0</td><td>98.6</td> <td>63.6</td><td>90.6</td><td>95.8</td> </tr>
   <tr> <td>HAL</td><td>SCAN_I2T</td> <td>78.3</td><td>96.3</td><td>98.5</td> <td>60.1</td><td>86.7</td><td>92.8</td> </tr>
   <tr> <td>DSRAN</td><td>BUTD, GRU</td> <td>76.3</td><td>94.9</td><td>98.4</td> <td>62.4</td><td>89.7</td><td>95.2</td> </tr>
   <tr> <td>DSRAN</td><td>BUTD, Bert</td> <td>77.1</td><td>95.3</td><td>98.1</td> <td>62.9</td><td>89.9</td><td>95.3</td> </tr>
   <tr> <td>DSRAN*</td><td>BUTD, GRU</td> <td>78.0</td><td>95.6</td><td>98.5</td> <td>64.2</td><td>90.4</td><td>95.8</td> </tr>
   <tr> <td>DSRAN*</td><td>BUTD, Bert</td> <td>78.3</td><td>95.7</td><td>98.4</td> <td>64.5</td><td>90.8</td><td>95.8</td> </tr>
   <tr> <td>GSMN</td><td>BUTD, sparse</td> <td>76.1</td><td>95.6</td><td>98.3</td> <td>60.4</td><td>88.7</td><td>95.0</td> </tr>
   <tr> <td>GSMN</td><td>BUTD, dense</td> <td>74.7</td><td>95.3</td><td>98.2</td> <td>60.3</td><td>88.5</td><td>94.6</td> </tr>
   <tr> <td>GSMN*</td><td>BUTD</td> <td>78.4</td><td>96.4</td><td>98.6</td> <td>63.3</td><td>90.1</td><td>95.7</td> </tr>
   <tr> <td>HAN</td><td>BUTD</td> <td>78.7</td><td>96.4</td><td>98.8</td> <td>65.4</td><td>90.5</td><td>95.3</td> </tr>
   <tr> <td>DIME</td><td>BUTD, I2T, Bert</td> <td>77.9</td><td>95.9</td><td>98.3</td> <td>63.0</td><td>90.5</td><td>96.2</td> </tr>
   <tr> <td>DIME</td><td>BUTD, T2I, Bert</td> <td>77.2</td><td>95.5</td><td>98.5</td> <td>62.3</td><td>90.2</td><td>95.8</td> </tr>
   <tr> <td>DIME*</td><td>BUTD, Bert</td> <td>78.8</td><td>96.3</td><td>98.7</td> <td>64.8</td><td>91.5</td><td>96.5</td> </tr>
   <tr> <td>SGRAF</td><td>BUTD, SAF</td> <td>76.1</td><td>95.4</td><td>98.3</td> <td>61.8</td><td>89.4</td><td>95.3</td> </tr>
   <tr> <td>SGRAF</td><td>BUTD, SGR</td> <td>78.0</td><td>95.8</td><td>98.2</td> <td>61.4</td><td>89.3</td><td>95.4</td> </tr>
   <tr> <td>SGRAF*</td><td>BUTD</td> <td>79.6</td><td>96.2</td><td>98.5</td> <td>63.2</td><td>90.7</td><td>96.1</td> </tr>
   <tr> <td>T-EMDE</td><td>BUTD, SAF</td> <td>78.3</td><td>95.7</td><td>98.5</td> <td>62.3</td><td>89.7</td><td>95.2</td> </tr>
   <tr> <td>T-EMDE</td><td>BUTD, SGR</td> <td>77.1</td><td>95.9</td><td>98.5</td> <td>61.6</td><td>89.5</td><td>95.1</td> </tr>
   <tr> <td>T-EMDE*</td><td>BUTD, SGRAF</td> <td>79.6</td><td>96.3</td><td>98.7</td> <td>63.5</td><td>90.4</td><td>95.6</td> </tr>
   <tr> <td>ACMM</td><td>BUTD</td> <td>81.9</td><td>98.0</td><td>99.3</td> <td>58.2</td><td>87.3</td><td>93.9</td> </tr>
   <tr> <td>ACMM*</td><td>BUTD</td> <td>84.1</td><td>97.8</td><td>99.4</td> <td>60.7</td><td>88.7</td><td>94.9</td> </tr>
   <tr> <td>PG*</td><td>BUTD, +GloVe</td> <td>84.0</td><td>95.8</td><td>97.8</td> <td>63.9</td><td>88.9</td><td>95.6</td> </tr>
   <tr> <td>SAN^</td><td>VggNet</td> <td>74.9</td><td>94.9</td><td>98.2</td> <td>60.8</td><td>90.3</td><td>95.7</td> </tr>
   <tr> <td>SAN^</td><td>ResNet</td> <td>85.4</td><td>97.5</td><td>99.0</td> <td>69.1</td><td>93.4</td><td>97.2</td> </tr>
   <tr> <td>GPO</td><td>IN, BiGRU</td> <td>76.5</td><td>95.3</td><td>98.5</td> <td>62.9</td><td>90.6</td><td>95.8</td> </tr>
   <tr> <td>GPO*</td><td>IN+VG, BiGRU</td> <td>80.0</td><td>97.0</td><td>99.0</td> <td>64.8</td><td>91.6</td><td>96.5</td> </tr>
   <tr> <td>GPO*</td><td>IN+VG, Bert</td> <td>82.2</td><td>97.5</td><td>99.5</td> <td>68.1</td><td>92.9</td><td>97.2</td> </tr>
   <tr> <td>GPO*</td><td>WSL, Bert</td> <td>85.6</td><td>98.0</td><td>99.4</td> <td>73.1</td><td>94.3</td><td>97.7</td> </tr>
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
   <tr> <td>TERN</td><td>BUTD, Bert</td> <td>38.4</td><td>69.5</td><td>81.3</td> <td>28.7</td><td>59.7</td><td>72.7</td> </tr>
   <tr> <td>Dual-Path</td><td>VggNet</td> <td>35.5</td><td>63.2</td><td>75.6</td> <td>21.0</td><td>47.5</td><td>60.9</td> </tr>
   <tr> <td>Dual-Path</td><td>ResNet</td> <td>41.2</td><td>70.5</td><td>81.1</td> <td>25.3</td><td>53.4</td><td>66.4</td> </tr>
   <tr> <td>VSE++</td><td>VggNet</td> <td>32.9</td><td>61.7</td><td>74.7</td> <td>24.1</td><td>52.8</td><td>66.2</td> </tr>
   <tr> <td>VSE++</td><td>ResNet</td> <td>41.3</td><td>71.1</td><td>81.2</td> <td>30.3</td><td>59.4</td><td>72.4</td> </tr>
   <tr> <td>GXN</td><td>ResNet</td> <td>42.0</td><td>--</td><td>84.7</td> <td>31.7</td><td>--</td><td>74.6</td> </tr>
   <tr> <td>SCO</td><td>VggNet</td> <td>40.2</td><td>70.1</td><td>81.3</td> <td>31.3</td><td>61.5</td><td>73.9</td> </tr>
   <tr> <td>SCO</td><td>ResNet</td> <td>42.8</td><td>72.3</td><td>83.0</td> <td>33.1</td><td>62.9</td><td>75.5</td> </tr>
   <tr> <td>CVSE++</td><td>ResNet</td> <td>43.2</td><td>73.5</td><td>84.1</td> <td>32.4</td><td>62.2</td><td>74.6</td> </tr>
   <tr> <td>DXR</td><td>ResNet, Bert</td> <td>44.9</td><td>75.2</td><td>84.7</td> <td>33.9</td><td>64.9</td><td>77.4</td> </tr>
   <tr> <td>PVSE</td><td>ResNet</td> <td>45.2</td><td>74.3</td><td>84.5</td> <td>32.4</td><td>63.0</td><td>75.0</td> </tr>
   <tr> <td>R-SCAN</td><td>BUTD, VrR-VG</td> <td>45.4</td><td>77.9</td><td>87.9</td> <td>36.2</td><td>65.5</td><td>76.7</td> </tr>
   <tr> <td>SAVE</td><td>ResNet</td> <td>46.7</td><td>76.3</td><td>86.1</td> <td>34.0</td><td>64.8</td><td>77.0</td> </tr>
   <tr> <td>MPL</td><td>SCAN_I2T</td> <td>46.9</td><td>77.7</td><td>87.6</td> <td>34.4</td><td>64.2</td><td>75.9</td> </tr>
   <tr> <td>GVSE*</td><td>BUTD</td> <td>47.2</td><td>76.6</td><td>88.4</td> <td>31.2</td><td>61.2</td><td>70.5</td> </tr>
   <tr> <td>CASC</td><td>ResNet</td> <td>47.2</td><td>78.3</td><td>87.4</td> <td>34.7</td><td>64.8</td><td>76.8</td> </tr>
   <tr> <td>OAN</td><td>BUTD</td> <td>47.8</td><td>81.2</td><td>90.4</td> <td>37.0</td><td>66.6</td><td>78.0</td> </tr>
   <tr> <td>MTFN</td><td>BUTD</td> <td>44.7</td><td>76.4</td><td>87.3</td> <td>33.1</td><td>64.7</td><td>76.1</td> </tr>
   <tr> <td>MTFN</td><td>BUTD, RR</td> <td>48.3</td><td>77.6</td><td>87.3</td> <td>35.9</td><td>66.1</td><td>76.1</td> </tr>
   <tr> <td>M3A</td><td>ResNet</td> <td>48.9</td><td>75.2</td><td>84.4</td> <td>38.3</td><td>65.7</td><td>76.9</td> </tr>
   <tr> <td>A3VSE</td><td>BUTD</td> <td>49.3</td><td>81.1</td><td>90.2</td> <td>39.0</td><td>68.0</td><td>80.1</td> </tr>
   <tr> <td>GVSE*</td><td>BUTD</td> <td>49.9</td><td>77.4</td><td>87.6</td> <td>38.4</td><td>68.5</td><td>79.7</td> </tr>
   <tr> <td>SGM</td><td>BUTD</td> <td>50.0</td><td>79.3</td><td>87.9</td> <td>35.3</td><td>64.9</td><td>76.5</td> </tr>
   <tr> <td>CAMP</td><td>BUTD</td> <td>50.1</td><td>82.1</td><td>89.7</td> <td>39.0</td><td>68.9</td><td>80.2</td> </tr>   
   <tr> <td>SCAN</td><td>BUTD, I2T_LSE</td> <td>46.4</td><td>77.4</td><td>87.2</td> <td>34.4</td><td>63.7</td><td>75.7</td> </tr>
   <tr> <td>SCAN*</td><td>BUTD, AVE+LSE</td> <td>50.4</td><td>82.2</td><td>90.0</td> <td>38.6</td><td>69.3</td><td>80.4</td> </tr>
   <tr> <td>GOT</td><td>SCAN_I2T</td> <td>50.5</td><td>80.2</td><td>89.8</td> <td>38.1</td><td>66.8</td><td>78.5</td> </tr>
   <tr> <td>PFAN*</td><td>BUTD</td> <td>50.8</td><td>83.9</td><td>89.1</td> <td>39.5</td><td>69.5</td><td>80.8</td> </tr>
   <tr> <td>PFAN++*</td><td>BUTD</td> <td>51.2</td><td>84.3</td><td>89.2</td> <td>41.4</td><td>70.9</td><td>79.0</td> </tr>
   <tr> <td>HOAD</td><td>BUTD</td> <td>51.2</td><td>81.7</td><td>89.1</td> <td>39.4</td><td>72.5</td><td>84.1</td> </tr>
   <tr> <td>HOAD</td><td>BUTD, +Dist</td> <td>51.4</td><td>81.8</td><td>89.1</td> <td>40.5</td><td>73.5</td><td>84.1</td> </tr>
   <tr> <td>CAAN</td><td>BUTD</td> <td>52.5</td><td>83.3</td><td>90.9</td> <td>41.2</td><td>70.3</td><td>82.9</td> </tr>
   <tr> <td>VSRN*</td><td>BUTD</td> <td>53.0</td><td>81.1</td><td>89.4</td> <td>40.5</td><td>70.6</td><td>81.1</td> </tr>
   <tr> <td>CCRS*</td><td>BUTD, SCAN</td> <td>47.9</td><td>78.1</td><td>88.2</td> <td>36.9</td><td>66.9</td><td>78.4</td> </tr>
   <tr> <td>CCRS*</td><td>BUTD, BFAN</td> <td>53.1</td><td>81.8</td><td>90.2</td> <td>38.3</td><td>67.8</td><td>78.6</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Image</td> <td>53.2</td><td>82.5</td><td>90.4</td> <td>38.9</td><td>68.5</td><td>79.2</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Text</td> <td>52.0</td><td>81.8</td><td>90.1</td> <td>38.6</td><td>68.1</td><td>79.1</td> </tr>
   <tr> <td>IMRAM</td><td>BUTD, Full</td> <td>53.7</td><td>83.2</td><td>91.0</td> <td>39.7</td><td>69.1</td><td>79.8</td> </tr>
   <tr> <td>MMCA</td><td>BUTD, Bert</td> <td>54.0</td><td>82.5</td><td>90.7</td> <td>38.7</td><td>69.7</td><td>80.8</td> </tr>
   <tr> <td>SMFEA</td><td>BUTD</td> <td>54.2</td><td>--</td><td>89.9</td> <td>41.9</td><td>--</td><td>83.7</td> </tr>
   <tr> <td>CAMERA</td><td>BUTD, Bert</td> <td>53.1</td><td>81.3</td><td>89.8</td> <td>39.0</td><td>70.5</td><td>81.5</td> </tr>
   <tr> <td>CAMERA*</td><td>BUTD, Bert</td> <td>55.1</td><td>82.9</td><td>91.2</td> <td>40.5</td><td>71.7</td><td>82.5</td> </tr>
   <tr> <td>DSRAN</td><td>BUTD, GRU</td> <td>51.9</td><td>81.6</td><td>89.8</td> <td>39.5</td><td>70.6</td><td>81.0</td> </tr>
   <tr> <td>DSRAN</td><td>BUTD, Bert</td> <td>53.7</td><td>82.1</td><td>89.9</td> <td>40.3</td><td>70.9</td><td>81.3</td> </tr>
   <tr> <td>DSRAN*</td><td>BUTD, GRU</td> <td>54.4</td><td>83.5</td><td>91.3</td> <td>41.5</td><td>71.9</td><td>82.1</td> </tr>
   <tr> <td>DSRAN*</td><td>BUTD, Bert</td> <td>55.3</td><td>83.5</td><td>90.9</td> <td>41.7</td><td>72.7</td><td>82.8</td> </tr>
   <tr> <td>TERAN</td><td>BUTD, Bert</td> <td>55.6</td><td>83.9</td><td>91.6</td> <td>42.6</td><td>72.5</td><td>82.9</td> </tr>
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
   <tr> <td>T-EMDE</td><td>BUTD, SAF</td> <td>56.7</td><td>--</td><td>90.7</td> <td>40.3</td><td>--</td><td>80.4</td> </tr>
   <tr> <td>T-EMDE</td><td>BUTD, SGR</td> <td>57.0</td><td>--</td><td>91.0</td> <td>40.0</td><td>--</td><td>80.1</td> </tr>
   <tr> <td>T-EMDE*</td><td>BUTD, SGRAF</td> <td>59.1</td><td>--</td><td>91.8</td> <td>41.8</td><td>--</td><td>81.7</td> </tr>
   <tr> <td>DIME</td><td>BUTD, I2T, Bert</td> <td>56.1</td><td>83.2</td><td>91.1</td> <td>40.2</td><td>70.7</td><td>81.4</td> </tr>
   <tr> <td>DIME</td><td>BUTD, T2I, Bert</td> <td>55.3</td><td>82.4</td><td>90.2</td> <td>39.7</td><td>70.3</td><td>81.0</td> </tr>
   <tr> <td>DIME*</td><td>BUTD, Bert</td> <td>59.3</td><td>85.4</td><td>91.9</td> <td>43.1</td><td>73.0</td><td>83.1</td> </tr>
   <tr> <td>SAN^</td><td>ResNet</td> <td>65.4</td><td>89.4</td><td>94.8</td> <td>46.2</td><td>77.4</td><td>86.6</td> </tr>
   <tr> <td>ACMM</td><td>BUTD</td> <td>63.5</td><td>88.0</td><td>93.6</td> <td>36.7</td><td>65.1</td><td>76.7</td> </tr>
   <tr> <td>ACMM*</td><td>BUTD</td> <td>66.9</td><td>89.6</td><td>94.9</td> <td>39.5</td><td>69.6</td><td>81.1</td> </tr>
   <tr> <td>GPO</td><td>IN, BiGRU</td> <td>55.1</td><td>81.9</td><td>89.9</td> <td>40.9</td><td>70.6</td><td>81.5</td> </tr>
   <tr> <td>GPO*</td><td>IN+VG, BiGRU</td> <td>59.8</td><td>86.1</td><td>92.8</td> <td>42.7</td><td>72.8</td><td>83.3</td> </tr>
   <tr> <td>GPO*</td><td>IN+VG, Bert</td> <td>62.5</td><td>87.8</td><td>94.0</td> <td>46.0</td><td>75.8</td><td>85.7</td> </tr>
   <tr> <td>GPO*</td><td>WSL, Bert</td> <td>68.1</td><td>90.2</td><td>95.2</td> <td>52.7</td><td>80.2</td><td>88.3</td> </tr>
   <tr> <td>PG*</td><td>BUTD, +GloVe</td> <td>68.7</td><td>88.7</td><td>93.0</td> <td>46.2</td><td>77.8</td><td>85.5</td> </tr>
</table> 

## ``Peformance of Identity-aware Datasets``

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
   <tr> <td>MCCL</td><td>MobileNet, CL</td> <td>48.21</td><td>--</td><td>78.27</td> </tr>
   <tr> <td>MCCL</td><td>MobileNet</td> <td>50.58</td><td>--</td><td>79.06</td> </tr>
   <tr> <td>MIA</td><td>VggNet</td> <td>48.00</td><td>70.70</td><td>79.30</td> </tr>
   <tr> <td>MIA</td><td>ResNet</td> <td>53.10</td><td>75.00</td><td>82.90</td> </tr>
   <tr> <td>A-GANet</td><td>ResNet</td> <td>53.14</td><td>74.03</td><td>82.95</td> </tr>
   <tr> <td>PMA</td><td>VggNet</td> <td>47.02</td><td>68.54</td><td>78.06</td> </tr>
   <tr> <td>PMA</td><td>ResNet</td> <td>53.81</td><td>73.54</td><td>81.23</td> </tr>
   <tr> <td>TIMAM</td><td>ResNet, Bert</td> <td>54.51</td><td>77.56</td><td>84.78</td> </tr>
   <tr> <td>CMAAM</td><td>MobileNet</td> <td>55.13</td><td>76.14</td><td>83.77</td> </tr>
   <tr> <td>ITMeetsAL</td><td>MobileNet</td> <td>51.85</td><td>73.36</td><td>81.27</td> </tr>
   <tr> <td>ITMeetsAL</td><td>ResNet</td> <td>55.72</td><td>76.15</td><td>84.26</td> </tr>
   <tr> <td>ViTAA</td><td>ResNet</td> <td>55.97</td><td>75.84</td><td>83.52</td> </tr>
   <tr> <td>FTD</td><td>ResNet</td> <td>57.84</td><td>78.33</td><td>85.43</td> </tr>
   <tr> <td>MGEL</td><td>VggNet</td> <td>52.68</td><td>74.37</td><td>83.11</td> </tr>
   <tr> <td>MGEL</td><td>MobileNet</td> <td>59.21</td><td>79.16</td><td>85.88</td> </tr>
   <tr> <td>MGEL</td><td>ResNet</td> <td>60.27</td><td>80.01</td><td>86.74</td> </tr>
   <tr> <td>SSAN</td><td>VggNet</td> <td>55.52</td><td>76.17</td><td>83.45</td> </tr>
   <tr> <td>SSAN</td><td>ResNet</td> <td>61.37</td><td>80.15</td><td>86.73</td> </tr>
   <tr> <td>NAFS</td><td>ResNet, Bert</td> <td>59.94</td><td>79.86</td><td>86.70</td> </tr>
   <tr> <td>NAFS</td><td>+RVN</td> <td>61.50</td><td>81.19</td><td>87.51</td> </tr>
</table> 

### *Performance of ICFG-PEDES*
<table>
   <tr> <td rowspan="2">Method_name</td> <td rowspan="2", align="center">Concise_note</td> 
        <td colspan="3", align="center">Text-to-Image</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td></tr>
   <tr> <td>Dual-Path</td><td>ResNet</td> <td>38.99</td><td>59.44</td><td>68.41</td> </tr>
   <tr> <td>CMPL</td><td>ResNet</td> <td>43.51</td><td>65.44</td><td>74.26</td> </tr>
   <tr> <td>MIA</td><td>ResNet</td> <td>46.49</td><td>67.14</td><td>75.18</td> </tr>
   <tr> <td>SCAN</td><td>ResNet</td> <td>50.05</td><td>69.65</td><td>77.21</td> </tr>
   <tr> <td>ViTAA</td><td>ResNet</td> <td>50.98</td><td>68.79</td><td>75.78</td> </tr>
   <tr> <td>SSAN</td><td>ResNet</td> <td>54.23</td><td>72.63</td><td>79.53</td> </tr>
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
