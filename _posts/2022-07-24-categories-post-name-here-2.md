---
title: "Faster R CNN"
excerpt: "Faster R CNN 논문 리뷰"

categories:
  - Paper_Review
tags:
  - [Faster_Rcnn, Object_Detection]

permalink: /categories/paper_review/Faster_Rcnn/

toc: true
toc_sticky: true

date: 2025-02-01
last_modified_at: 2025-02-01
---
# Faster R-CNN

작성일시: 2024년 12월 27일 오후 8:40
강의 번호: CV Lab
복습: No

## 개념정리

### 1-Stage Detector , 2-Stage Detector

![image.png](/assets/images/faster_rcnn/image.png)

- <2-Stage Detecor> - Regional Proposal 과 Classification이 순차적으로 이루어진다.
    
    ![image.png](/assets/images/faster_rcnn/image1.png)
    
    - Regional Propoal(RP)
        - 기존 이미지에서는 object detection을 위해서 sliding window방식을 이용했습니다.
            - sliding window방식은 이미지에서 모든 영역을 다양한 크기의 window로 탐색하는 것
        - 비효율성을 개선하기 위해 물체가 있을만한 영역을 빠르게 찾아내는 알고리즘이 Region Proposal
            - ex) Selective Search, Region Proposal Network, Edge Boxes
        
        **Sliding window**
        
        장점 :
        
        단점 : 여러개의 사이즈를 이용하여 찾아야 다양한 크기의 객체를 찾을 수 있어 **모든 경우의 수를 따져야하여 수행시간이 오래걸리고 적절한 검출이 어렵다.**
        
    - Selective Search
        
        Selective Search의 기본 아이디어:
        
        1. 이미지를 작은 영역(세그먼트)들로 먼저 나눕니다.
        2. 비슷한 특성(색상, 질감, 크기 등)을 가진 세그먼트들을 점진적으로 병합합니다.
            1. Greedy algorithm 사용 : 현재상태에서 가장 유리한 것을 선택
        3. 병합 과정에서 다양한 크기와 모양의 후보 영역(Region Proposal)들을 생성합니다.
        4. 이 후보 영역들 중에서 객체일 가능성이 높은 영역을 선택합니다.
        
        장점 : 
        
        - 효율성 : 전체가 아니라 특정 영역만 검사하기 때문에 계산 효율성 상승
        - 다양성 : 다양한 스케일과 모양의 물체 탐색가능, 색,크기,질감등 다양한 특징 고려때문
        
        단점 : **selective search를 사용하면 end-to-end로 학습이 불가능하고, 실시간 적용에도 어려움이 있음**
        
    
    ![image.png](/assets/images/faster_rcnn/image2.png)
    
    - ROI(Region of Interest)
        - Selective Search(선택적 검색) 알고리즘을 사용하여 약 2000개의 ROI 후보 영역을 생성합니다. 이 과정은 다음과 같습니다:
        1. 입력 이미지에 대해 초기 세그멘테이션(Segmentation)을 수행합니다.
        2. 색상(Color), 질감(Texture), 크기(Size), 형태(Shape) 등의 유사도를 기반으로 작은 영역들을 계층적으로 그룹화합니다.
        3. 이렇게 생성된 각각의 ROI는 서로 다른 크기를 가지고 있어, 고정된 크기(227x227)로 워핑(Warping)됩니다.
        - 각 ROI는 CNN(Convolutional Neural Network)을 통과하여 특징(Feature)을 추출하게 됩니다. 추출된 특징은 SVM(Support Vector Machine)분류기를 통해 객체의 클래스를 분류하고, 바운딩 박스(Bounding Box) 회귀기를 통해 객체의 정확한 위치를 예측합니다.
        - 하지만 RCNN의 ROI 처리 방식에는 몇 가지 한계가 있습니다:
        1. 각 ROI를 개별적으로 처리하기 때문에 연산 속도가 매우 느립니다.
        2. 학습 과정이 다단계로 이루어져 복잡합니다.
        3. 실시간 객체 탐지가 어렵습니다.
    
    ![image.png](/assets/images/faster_rcnn/image3.png)
    
    - 2- stage detector 동작 과정
        
    ![image.png](/assets/images/faster_rcnn/image4.png)
        
        Selective search, Region proposal network와 같은 알고리즘을 및 네트워크를 통해 **object가 있을만한 영역을 우선 뽑아낸다.** 이 영역을 **RoI**(Region of Interest)라고 한다. 이런 영역들을 우선 뽑아내고 나면 각 영역들을 convolution network를 통해 classification, box regression(localization)을 수행한다.
        
- <1-Stage Detector>
    - Regional proposal와 classification이 동시에 이루어진다.
    - 즉, classification과 localization문제를 동시에 해결하는 방법이다.

![image.png](/assets/images/faster_rcnn/image5.png)   

따라서 1-stage detector는 비교적 빠르지만 정확도가 낮고

배경이 객체보다 많아 학습이 불균형하고, 세밀한 특징 추출이 어려워 정확도가 낮습니다

2-stage detector는 비교적 느리지만 정확도가 높다.

구조가 복잡해서 느리지만 자세한 분석이 가능하여 정확도가 높다

### 목표 Task 설명

![image.png](/assets/images/faster_rcnn/image6.png)   

- **Classification :** Single object에 대해서 object의 클래스를 분류하는 문제이다.
- **Classification + Localization :** Single object에 대해서 object의 위치를 bounding box로 찾고 **(Localization)** +클래스를 분류하는 문제이다. **(Classification)**
    - Localization : 어디있는지 찾는 것
- **Object Detection :** Multiple objects에서 각각의 object에 대해 Classification + Localization을 수행하는 것이다.
- **Instance Segmentation :** Object Detection과 유사하지만, 다른점은 object의 위치를bounding box가 아닌 실제 edge로 찾는 것이다.

[hog]

![image.png](/assets/images/faster_rcnn/image7.png)

보행자 검출을 위한 HOG는 기본적으로 64 x 128 크기의 영상에서 계산한다.

HOG 알고리즘은 먼저 입력 영상으로부터 그래디언트를 계산한다.

그래디언트는 크기와 방향 성분으로 계산하며, 방향 성분은 0° 부터 180°까지 범위로 설정한다.

그 다음 입력 영상을 8 x 8 크기로 분할한다. 입력 영상인 64 x 128 영상에서는 가로 방향으로 8개, 세로 방향으로 16개가 생성된다.

이때 8 x 8 부분 영상을 **셀(cell)**이라고 하며, 인접한 4개의 셀을 합쳐서 **블록(block)**이라고 한다.

각각의 셀로부터 그래디언트 방향 성분에 대한 히스토그램을 구하며, 이때 방향 성분을 20°단위로 구분하면

총 9개의 빈으로 구성된 방향 히스토그램이 만들어진다.

[dpm]

만약 템플릿이 얼굴 필터라면, 블록을 돌아다니면서 어디가 얼굴인지 svm을 통해 classification을 하게 된다.

결국 다양한 템플릿 필터를 통해  svm으로 classification 한 결과를 이용하여 최종적인 분류를 한 것이니 앙상블 기법이 적용되었다고 할 수 있다.

하지만 DPM에는 문제점이 있었다.

슬라이딩 윈도우를 통해 이미지의 모든 부분을 일일이 계산하면 굉장히 많은 바운딩 박스가 형성될 것이고 각각의 바운딩 박스마다 복잡한 classification 과정을 거치게 된다. 이렇게 되면 background를 검출하는 잉여시간이 너무 많아진다.

## [R - CNN]

: 객체 검출을 위해 제안된 딥러닝 모델, 여러개의 후보영역(Region Proposal)로 나눈 뒤 각각의 영역을 CNN에 입력하여 특징을 추출한 후, 분류와 경계박스 좌표를 예측하는 방법. Image classification역할 CNN + localization역할 Regional proposal

### Introduction

1. PASCAL VOC 2012에서 기존 최고 성능 대비 50% 이상 향상된 mAP 62.4%달성
    1. Pascal Voc : 20개 객체 범주에 대해 바운딩 박스와 클래스 라벨이 주어짐
2. ILSVRC 2013 detection task에서 mAP 31.4% 달성하여 2위인 OverFeat(24.3%) 대비 큰 향상
    1. ILSVRC : imageNet 데이터셋 기반 컴비 경진대회
    2. mAP : mean Average Precison. 얼마나 정확하고 재현율이 좋은지 종합적 평가, 전체적으로 고른 성능을 내는 모델일수록 mAP가 높게 나온다.
    
    ![image.png](/assets/images/faster_rcnn/image8.png)
    

[abstact]

![image.png](/assets/images/faster_rcnn/image9.png)

[문제점 2가지]


1. Deep network를 이용해서 object를 어떻게 localize할 수 있을까?(=bound Box 어떻게 그리지?)
    1. 접근법 1 : localizing을 regression문제로 접근
        
        장점 : 단일 object localizing 성능 좋음
        
        단점 : 다중 object localizing 성능 낮음
        
    2. 접근법 2: sliding-window 기법 적용
        
        장점 : computeational effciency 
        
        단점 : 모든 object에 동일한 aspect ratio를 사용 >> 다양한 크기 반양 못함
        
    
    결론 >> c. Recognition Using Regions를 사용하자!!
    
2. 적은 양의 label 된 데이터로 high - capacity 모델을 어떻게 학습할수 있을까?
    1. 이전 사용법 : unsupervised pre-training(with supervised Fine-tuning)
    
    >> supervised pre training, 대량의 데이터셋(ILSVRC)에서 training
    
    (pascal voc coco등은 image net보다 규모가 훨씬 작다)
    

> R - CNN 프로세스
> 
> 
> ![image.png](/assets/images/faster_rcnn/image10.png)
> 
> ![image.png](/assets/images/faster_rcnn/image11.png)
> 
1. Input Image(이미지 입력)
2. Extract Region Proposals (~2k) : (extracts around 2000 bottom-up region proposals)
    - 물체가 있을 법한 영역을 찾는다.
    - 기존의 sliding Window방식 해결 >> Selective Search 방식 사용
    
    ![image.png](/assets/images/faster_rcnn/image12.png)
    
    - 2천개의 후보영역을 추출해서 이들 각각을 객체 검출에 활용합니다.
    - Bottom-up방식(주변 픽셀사이의 유사도, 엣지, 색상, 질감등 종합하여 후보영역을 합치고 쪼개면서 추출, Selective Search가 대표적 예시)
    - Selective Search : 여러 덩어리를 만든 다음, 비슷한 것 끼리 계속 병합하여 영역을 확장해가며 다양한 크기의 후보영역 생성.
    - low픽셀 레벨(저수준 정보)부터 후보영역을 쌓아 올리는 과정이라서 Bottom up 과정이라고 부름
    
    ![image.png](/assets/images/faster_rcnn/image13.png)
    
    Q. 2천개인 이유
    
    - 너무 많은 후보영역 : 연산량 폭증
    - 너무 적은 후보영역 : 놓치는 물체가 생김
    - CNN네트워크 입력 사이즈에 맞게 Region Proposal output 크기를 동일 input size로 WARP
        - WARP : 이미지,영상을 기하학적으로 변형하는 처리기술
            - Transform Func를 사용하여 픽셀의 새로운 위치를 계산하며
            - (x,y) 좌표를 (x' , y')좌표로 대응시키는 작업
        - 동일한 input size를 만드는 이유 :
            1. Convolution Layer에는 input size가 고정이지 않는다.
            2. 마지막 FC Layer에서 input size는 고정이기 때문에 CV에 대한 output size 동일해야함
    1. 2000개의 Warped Image를 각각 CNN 모델에 넣는다.
        1. 고정된 크기의 output을 얻기 위해 warp 작업을 통해서 동일한 input size를 생성후 투입
        2. 2000개의 후보 영역을 Fine Tune된 Alexnet에 입력하여 2000(후보영역의 수) * 4096(feature vector의 차원)크기의 Feature Vector를 추출합니다
    2. Fine tuning pre-Trained Alexnet
        1. 입력으로 들어온 Region Proposal은 객체를 포함할 수도 있으며, 배경을 포함할 수도 있다.
        2. 예측 객체수 n개라고 할때 배경을 포함하여(n+1)개의 class를 예측하도록 설계
        3. 객체와 배경을 모두 포함한 학습데이터를 구성 
    3. 각각의 Conv결과에 대해 Classification을 진행하여 결과를 얻는다.
        1. SVM을 사용함
        2. softmax 사용하지 않은 이유 시기상 데이터가 많지 않았기 때문
        
        ![image.png](/assets/images/faster_rcnn/image14.png)
        
        <SVM> support Vector Machine
        
        1. 지도학습 알고리즘
        2. 최적의 결정 경계(선)을 찾는 것
        3. 마진을 최대화 (의미 : 구분하는 선과 support vector와의 거리)
        
        ![image.png](/assets/images/faster_rcnn/image15.png)
        

### 3. Module Design

### 3.1.1 Region Proposals

- **region proposals를 생성하는 방법들은 다양**함 : objectness, **selective search**, category-independent object proposals, CPMC(constrained parametric min-cuts), multi-scale combinatorial grouping 등등...

- **R-CNN**은 이러한 방법들 중 어떤 것으로도 region proposal를 생성할 수 있지만, 이전 연구 [참고 문헌 21, 54]와 비교하기 위하여 **region proposal 생성**에 **Selective Search를 택함**.

### 3.1.2 Feature Extraciton

- **CNN**을 사용하여 **각 region proposal**에서 **고정된 길이의 feature vector**를 생성
- CNN 아키텍쳐로 **TorontoNet(=Alexnet)(input 사이즈 : 227 pixel)**과 **OxfordNet(16개의 layer, input 사이즈 : 224 pixel)**를 사용
- 둘 다 **feature vector는 4096 차원**
    - 너무 적으면 충분한 표현능력 갖추지 못하고
    - 너무 크면 과적합 문제가 생긴다
    - 경험적으로 좋은 성능과 안정성 간의 절충안
- **region을 input size에 맞추기** 위해 모든 pixel를 **wrap**(resize)

![image.png](/assets/images/faster_rcnn/image16.png)

### IoU(intersction over union)

![image.png](/assets/images/faster_rcnn/image17.png)

### B box regression

Region Proposal 과정에서 Selective Search 알고리즘을 활용해 제안된 영역은 실제 객체 영역과 차이가 있습니다. **Bounding Box regressor은 region proposal에서 제안된 이미지 영역을 실제 객체 영역에 맞도록 조정해주는 역할**을 합니다.

아무래도 완전히 정확하지는 않기 때문에 물체를 정확히 감싸도록 조정해주는 **선형회귀 모델(Bounding Box Regression)**을 넣었습니다.

![image.png](/assets/images/faster_rcnn/image18.png)

![image.png](/assets/images/faster_rcnn/image19.png)

![image.png](/assets/images/faster_rcnn/image20.png)

B Box를 적용 한 것이 더욱더 정확도가 높았습니다.

### <R - CNN 문제점>

1. 오래걸린다
    1. 2000개의 영역들에 대해서 모두 CNN을 통과함
        
        Training Time : 84시간
        
        Testing Time(gpu K40) : Frame 당 13s
        
        Testing Time(cpu) : Frame 당 53s
        
    2. Selective Search가 cpu를 사용하기 때문도 있다.
2. 복잡하다
    1. CNN, SVM,BBox Regression까지 3가지의 모델을 필요로하는 복잡한 구조 

## <FAST R - CNN>

### Introduction

![image.png](/assets/images/faster_rcnn/image21.png)

- Fast R cnn은 1장의 사진만 입력받음
- Region Proposals의 크기를 warp 안해도 된다
- ROI(Region of Interest) Pooing을 통해 고정된 크기의 Feature Vector를 FC Layer에 전달
- Multi - Task Loss를 사용하여 모델을 한번에 학습시킨다.

1) 하나의 CNN을 통해 전체 이미지를 한 번만 처리

- 전통적인 R-CNN은 후보영역(region proposal) 각각에 대해 CNN을 돌려 특징을 얻어야 했으므로 연산량이 매우 컸습니다.
- 반면 SPP-net과 Fast R-CNN은 이미지 전체를 단 한 번만 CNN에 통과시켜 특징 맵을 추출한 후, 필요한 부분영역만을 골라서 쓴다는 점이 핵심입니다.

2) 특징 맵에서 후보영역에 해당하는 부분을 추출

- 실제 분류나 회귀(regression)에 필요한 것은 물체가 존재할 것으로 예측되는 후보영역이므로, CNN이 추출한 특징 맵에서 해당 영역을 잘라냅니다.
- SPP-net에서는 Spatial Pyramid Pooling을, Fast R-CNN에서는 RoI Pooling을 이용해 고정된 크기의 특징 벡터로 변환하여 분류 및 박스 회귀에 활용합니다.

3) 영역 단위 분류(또는 회귀)

- 후보영역에서 잘라낸 특징들을 입력으로 하여, 분류(layer)와 BBox 회귀(layer)를 통해 각 영역에 대해 어떤 클래스인지, 그리고 그 경계(Bounding Box)는 어떻게 조정해야 하는지를 예측합니다.

4) 복잡도와 속도

- R-CNN은 각 후보영역마다 CNN을 실행하므로 연산량이 매우 컸지만, SPP-net과 Fast R-CNN은 이미지 전체를 한 번만 CNN으로 처리하고 이후 단계에서 후보영역별 연산을 수행하므로, 전체 연산량이 크게 줄어듭니다.
- 결과적으로 R-CNN 대비 약 160배 정도 빠른 속도를 달성하며, 대략적인 복잡도 역시 R-CNN에 비해 크게 개선되었습니다(슬라이드에 표기된 대로 약 600×1000×1 단위의 연산량으로 표시됨).

### ROI Pooling Layer

: feature map에서 region proposals에 해당하는 ROI를 지정한 크기의 grid로 나눈 후 max pooling진행. 

>> 고정된 크기의 feature map을 출력 가능

![image.png](/assets/images/faster_rcnn/image22.png)

1. 원본 이미지를 CNN모델 통과시켜 Feature map을 얻는다.
2. R-CNN과 다르게 VGG를 사용합니다        
    1. 800 * 800이미지를 VGG를 이용해서 8*8 Feature map을 얻음.(sub sampling ratio(=pooling) = 1/100 )
    2. VGG 사용하는 이유 : 단순하고 효율적인 구조를 가짐(3*3 conv 필터 사용)
    3. 3*3 conv 필터 사용의 이점 : 큰 필터 한번 사용하는 것보다 파라미터수 적게
3. 원본에서 Selective Search를 이용해서 Region proposal 얻음
    1. 500 * 700 크기의 Region proposal 얻음
4. Feature map에서 각 region proposals에 해당하는 영역을 추출.ROI projection을 이용
    1. Region Proposal의 중심점 좌표,W,H와 sub sampling ratio 활용>feature map 투영
    2. feature map에서 region proposal에 해당하는 5*7 영역 추출
5. ROI feature map을 지정한 sub window크기에 맞게 grid로 나눠줍니다
    1. 2*2크기에 맞게 grid를 나눠줍니다
6. grid의 각 셀마다 max pooling을 수행하여 고정된 크기의 feature map을 얻는다
    1. 각 grid셀 마다 max pooling을 수행하여 2*2 feature map을 얻는다.

>> 미리 지정한 크기의 sub window에서 max pooling을 수행하여 **region proposal의 크기가 서로 달라도 고정된 크기의 feature map을 얻을 수 있다.**

### Multi-Task Loss

: Feature Vector를 Classifier와 BBox Regressior를 동시에 학습 시킬 수 있다.

각각의 ROI에 대하여 Multi task loss를 사용하여 학습하며 두모델을 한번에 학습.

$$
L(p, u, t^u, v) = L_{cls}(p, u) + \lambda[u \ge 1]L_{loc}(t^u, v)
$$

$p = (p0, ....., p_k)$ : (K+1)개의 Class Score

$u$ : Ground truth class score

$t^u = (t_x^u, t_y^u, t_w^u, t_h^u)$ : 예측한 Bounding box 좌표를 조정하는 값

$v = (v_x, v_y, v_w, v_h)$ : 실제 bounding box의 좌표값

$L_{cls}(p, u) = -log {p_u}$ : clssification loss(Log loss)

$L_loc(t^u, v) = \sum_{i \in \{x,y,w,h \}} smooth_{L_1}(t_i^u - v_i)$ : regression loss(smooth L1 loss)

$smooth_{L_1}(t_i^u - v_i) = \begin{cases} 0.5x^2,  if |x| < 1 \\ |x| - 0.5, otherwise \end{cases}$

$\lambda$ : 두 loss 사이의 가중치를 조정하는 balancing hyperparameter.

- K개의 class를 분류한다고할 때, 배경을 포함한 (K+1)개의 class에 대하여 Classifier를 학습시켜줘야 합니다.
- u는 positive sample인 경우 1, negative sample인 경우 0으로 설정되는 **index parameter**입니다.
- **L1 loss**는 R-CNN, SPPnets에서 사용한 L2 loss에 비행 outlier에 덜 민감하다는 장점이 있습니다.
- λ=1 로 사용합니다.
- multi task loss는 0.8~1.1% mAP를 상승시키는 효과가 있다고 합니다.

### Training Fast R - CNN

![image.png](/assets/images/faster_rcnn/image23.png)

![image.png](/assets/images/faster_rcnn/image24.png)

1. Initializing pre trained network
    1. Feature map 추출을 위해 VGG 16모델 사용 
    2. VGG 모델 마지막 max pooling layer를 ROI pooling layer로 대체.
    3. ROI pooling 을 통해 출력되는 feature map 크기를 FC layer와 호환되게 7*7설정
    4. 마지막 FC layer를 2개의 FC layer로 대체합니다 
    5. 첫번째 FC layer는 K개의 Class와 배경을 포함한 K+1개의 output을 가지는 classifier
    6. 두번째 FC layer는 각 Class별로 bounding box의 좌료를 조정하여 (k+1)*4개의 output을 가지는 bounding box regressor입니다
    7. Conv Layer3까지의 가중치 값은 고정시키고 이후 layer까지의 가중치 값이 학습될 수 있도록 fine tuning합니다.
2. Region proposal by selective search
    1. 2000개의 region proposals를 추출
3. Feature extraction by VGG 16
    
    VGG16 모델에 224x224x3 크기의 원본 이미지를 입력하고, layer13까지의 feature map을 추출합니다
    
    - **Input** : 224x224x3 sized image
    - **Process** : feature extraction by VGG16
    - **Output** : 14x14x512 feature maps
4. Max pooling By ROI pooling
    1. RoI pooling layer는 VGG16의 마지막 pooling layer를 대체한 것입니다. 이 과정을 거쳐 고정된 7x7 크기의 feature map을 추출합니다. 
    - **Input** : 14x14 sized 512 feature maps, 2000 region proposals
    - **Process** : RoI pooling
    - **Output** : 7x7x512 feature maps
5. FC Layers
    
    다음으로 region proposal별로 7x7x512(=25088)의 feature map을 flatten한 후 fc layer에 입력하여 fc layer를 통해 4096 크기의 feature vector를 얻습니다.
    
    - **Input** : 7x7x512 sized feature map
    - **Process** : feature extraction by fc layers
    - **Output** : 4096 sized feature vector
6. Classifier
    1. 4096 크기의 feature vector를 K개의 class와 배경을 포함하여 (K+1)개의 output unit을 가진 fc layer에 입력합니다.
    - **Output** : (K+1) sized vector(class score)
7. B box Regressor
    1. 4096 크기의 feature vector를 class별로 bounding box의 좌표를 예측하도록 (K+1) x 4개의 output unit을 가진 fc layer에 입력합니다
    - **Output** : (K+1) x 4 sized vector
8. Multi - task loss
    1. Multi-task loss를 사용하여 하나의 region proposal에 대한 Classifier와 Bounding box regressor의 loss를 반환합니다. 이후 Backpropagation을 통해 두 모델(Classifier, Bounding box regressor)을 한 번에 학습시킵니다.

## <FASTER R - CNN>

### Abstract

1. 기존 객체 위치 추정을 위해서 region proposal(영역 추정)알고리즘 사용
2. 여전히 영역 추정단계에서 bottleneck(병목) 현상이 생기는 단점
    1. Region proposal이 독립적으로 존재하기 때문
3. 해결 방법으로 RPN(Region Proposal Network) 영역추정네트워크 기법을 제안
4. RPN은 객체 탐지 네트워크와 합성공 피처들을 함께 공유하여 비용이 거의 없음
5. RPN은 객체의 경계박스, 클래스 점수(객체 여부 점수화)를 동시에 예측하는 네트워크
6. end to end 훈련 가능 >> 품질 좋은 영역 추정 경계박스 생성 가능

### Introduction

1. Selective Search는 cpu에서 이미지당 2s로 느리다. 
2. EdgeBox를 활용하면 이미지당 0.2s, 여전히 느림
3. FAST R-CNN은 대부분 GPU 이용하지만 영역추정은 CPU에서 수행
4. Deep CNN으로 영역추정 하는 알고리즘을 통해 계산 비용을 거의 없앰
5. 객체탐지네트워크 + CNN(featuremap)을 공유하는 RPN으로 가능하게함
6. Test단계에서 CNN결과를 공유해 영역 추정속도를 크게 개선(이미지당 10ms)
7. RPN구성
    1. Fast R-CNN이 가지고 있는 region based detectors가 사용하는 CNN feature map에 몇가지 합성곱 layer만 더하면 RPN 구축
    2. 객체의 대략적 경계박스를 찾아주고 동시에 객체 존재 여부를 점수로 나타냄.
    3. 완벽한 Fully Convolutional Network(FCN)+ end to end Train 가능
8. (c) 앵커박스 방식
    
    ![image.png](/assets/images/faster_rcnn/image25.png)
    
    : CNN을 통과시킨 feature map에 대해 미리 정의된 여러 크기와 비율의 박스를 겹쳐 객체 가능성을 예측하는 방식, 다양한 스케일과 가로세로 비율을 갖는 이미지나 필터 사용하지 않아도 됌(=단일 스케일 이미지만 사용해도 되기 때문에 빠름)
    

### Related Work

- Object Proposals
    - slective serach, sliding window(edge Boxex)등이 있으나 객체 탐지 모델과 독립적 사용
- Deep Networks for Object detection
    - OverFeat기법 OR MultiBox기법 : 딥러닝을 사용해 경계박스 예측하는 방법>공유 X
    - 합성 곱 피처계산결과 공유 >> 효율성 정확도 상승
    - 

### Faster R - CNN

: 2가지 모델로 구성(1. 영역 추정을 위한 깊은 합성곱 네트워크 2. 객체 탐지 모듈)

- 전체 네트워크 : 객체 탐지를 위한 single, unified 구조
- 'attention'과 유사한 mechanisms

![image.png](/assets/images/faster_rcnn/image26.png)

### 3.1 Region Proposal Networks(RPN)

- RPN은 크기에 상관없이 이미지 전체를 입력 후 영역 추정 경계박스를 반환
- 각 경계박스는 객체가 있는지 여부를 점수로 나타냄
- Fully convolutional Network에서 process가 진행된다.

![image.png](/assets/images/faster_rcnn/image27.png)

1. 영역 제안을 생성하기 위해, 마지막 공유 합성곱 계층에서 나온 합성곱 특성 맵 위를 작은 네트워크가 슬라이딩하며 이동합니다.
2. 더 낮은 차원의 특징공간으로 매핑(ZF : 256차원, VGG : 512차원) + ReLU 활성함수.
3. 특징 벡터는 두 개의 형제 관계를 갖는 완전 연결 계층(상자 회귀 계층(reg)과 상자 분류 계층(cls))으로 전달됩니다
4. 위 논문에서는 n = 3을 사용하며 입력이미지 수용영역은 크다(ZF 171픽셀, VGG 288픽셀)
    1. n=3 의미 : 마지막 단계에서 3*3 슬라이딩윈도우 or 3*3합성곱 커널 의미
    2. 큰 수용영역을 사용하는 이유 : 주변 context까지 함께 고려가능, 심층 신경망은 초기에 작은 커널로 시작해도 층이 깊어질수록 수용영역이 누적되면서 최종적으로 넓게 인지 가능
5. n*n합성곱 계층 뒤에 두개의 1*1합성곱계층(reg,cls)이어지도록 구현이 자연스러운 형식

### 3.1.1 Anchors

각 슬라이딩 윈도우 위치마다 최대로 생성되는 최대 제안의 갯수를 K, 동시에 여러개 영역제안 예측

reg layer(회귀 계층) 좌표값은 4k개 : x, y, width, height

cls layer(분류 계층) 좌표값은 2k개 : 객체 / 비객체(background)

![image.png](/assets/images/faster_rcnn/image28.png)

k=9, 3가지 스케일(128 256 512), 3가지 비율(1:1 2:1 1:2).

feature map 크기 : W * H ~= 2400

: 2400이 나오는 이유는 짧은변 600, 긴변 1000일 때 vgg16경우 1/16으로 축소되기 때문

앵커 박스 갯수 : W * H * K = 2400 * 9 = 21,600

![image.png](/assets/images/faster_rcnn/image29.png)

- Stride는 합성곱 커널이 입력 위를 슬라이딩할 때 몇 픽셀씩 건너뛰는지를 나타내는 파라미터로, 이를 2로 설정하면 특성 맵의 가로/세로가 절반으로 줄어듭니다.
- 이러한 Downscaling을 통해 모델은 메모리와 연산량을 줄이면서, 더 깊은 특징을 학습하고 넓은 receptive field를 갖게 되어 객체 인식이나 분류 등의 작업을 더 잘 수행할 수 있습니다.

### <Translation - Invariant Anchors>

앵커 피라미드 방식(Pyramids of anchors) 

1. 다양한 크기의 앵커 박스를 활용하여 객체분류, 경계박스 회귀를 수행.  
2. 추가 연산 없이 피처를 공유할 수 있어 효율적

### 3.1.2 Loss Function

RPN훈련을 위해서 앵커박스 마다 이진분류 수행(객체가 있는지 없는지 여부)

분류를 위해서 앵커박스에 Positive label이 있어야함

1. Ground truth box(실제 경계 박스) + IoU가 가장 큰 앵커
2. Ground truth box(실제 경계 박스) + IoU가 0.7이 넘는 앵커

실제 경계 박스와 IoU가 높다면 여러 앵커 박스를 positive label로 간주하며, 2번 조건을 우선으로 찾으며 1번 조건으로 positive label 앵커 박스를 찾습니다.

Iou < 0.3일 경우에는 negative label로 간주합니다

![image.png](/assets/images/faster_rcnn/image30.png)

1. p_i는 i번째 앵커 박스가 객체일 확률
2. ground-truth label인 p*_i는 앵커가 positive이면 1, negative(=배경)이면 0입니다.
3.  t_i는 예측 경계 박스의 4가지 좌표값
4. t*_i는 실제 경계 박스의 4가지 좌표값
5. L_cls는 두 가지 클래스(객체 vs. 객체가 아님)에 대한 로그 손실
6.  L_reg는 경계 박스 회귀 손실을 뜻합니다. 참고로 회귀 손실값은 positive 앵커 박스일 때만(객체 일 때만) 활성화
7. N_cls 정규화 (256)
8. N_reg 정규화 (2400)
9.  λ 파라미터 : 분류,회기 손실 간 균형을 맞추기 위해서 (10)

![image.png](/assets/images/faster_rcnn/image31.png)

: 경계박스 회귀

### 3.1.3 Training RPNs

: RPN은 SGD와 역전파로 end to end 훈련을 합니다

여러 앵커박스를 만들어 영역 추정을 할 때 모든 앵커박스 마다 손실 함수를 적용하면 negative lable에 편향 된 결과를 냅니다.(positive 보다 negative label이 압도적으롬 낳기 때문(배경))

positive 앵커 negavie 앵커를 1:1 비율로 뽑아서 128개 128개가 도비니다.

훈련 시 평균 0, 분산 0.01인 가우시안 분포에서 뽑은 가중치로 초기화

### 3.2 Sharing Features for RPN and Fast R - CNN

Alternating training으로 4단계의 훈련 알고리즘을 채택

1. RPN을 훈련, 이미지넷으로 pre trained 모델로 초기화 하고 영역 추정 작업을 위해 end to end 파인튜닝을 합니다.
2. 첫 번째 단계에서 생성된 영역 추정 경계박스를 활용해 독립적인 Fast R cnn 훈련
3. RPN 훈련 초기화를 위해 Fast r cnn 사용, 오직 RPN 계층만 파인튜닝 및 합성곱 계층 공유
4. 공유된 합성곱 계층 고정 후 Fast R cnn 계층 파인튜닝

### 3.3 Implementations Details

3가지 스케일 : 128², 256², 512² 픽셀

비율 : 1:1, 1:2, 2:1

![image.png](/assets/images/faster_rcnn/image32.png)

>> RPN 사용 결과

![image.png](/assets/images/faster_rcnn/image33.png)

>> ZF-net 사용시 평균 proposal size

: 앵커 박스 크기보다 큰 객체도 찾으며 . 객체를 찾기 위해 앵커 박스가 꼭 객체 전체를 감싸야만 할 필요는 없다는 점을 말해줌.

<비최댓값 억제(Non-Maximum Suppression, NMS)>

: 경계 박스 가운데 가장 확실한 경계 박스만 남기고 나머지 경계 박스는 제거하는 기법입니다. 

이미지 경계를 가로지르는 앵커 박스(cross-boundary anchors) 앵커 박스를 제거하지 않으면 훈련 속도도 느리고 성능도 좋지 않다.

RPN에서 만든 영영 추정 경계박스는 겹치는 영역이 많아 NMS방법을 이용하여 겹치는 영역 제거

NMS IoU 임계값 0.7 설정. 경계박스가 6천개에서 2천개로 줄어듬

경계박스가 줄어듬에 따라 속도가 빨라지며 성능이그대로

이유 : 제거 되는 박스는 중복 박스이기 때문에 유의미한 박스만 남기때문

### Experiments

### 4.1 Experiments on Pascal Voc

![image.png](/assets/images/faster_rcnn/image34.png)

PASCAL VOC 2007은 훈련/검증 이미지 5,000개와 테스트 이미지 5,000개

Faster R-CNN은 mAP 59.9%를 달성했습니다. Faster R-CNN은 테스트 단계에서는 영역 추정 경계 박스 개수가 최대 300개입니다. 

NMS를 적용해서 더 적은 경계 박스를 남길 수도 있습니다. 

Faster R-CNN은 피처를 공유하고, 테스트 단계에서 경계 박스 개수도 적기 때문에 선택적 탐색(SS)이나 EdgeBoxes(EB)에 비해 더 빠릅니다.

**<Performance of VGG-16>**

RPN 훈련을 ZF가 아닌 VGG - 16으로 훈련. Test는 Voc 2007로만 함

![image.png](/assets/images/faster_rcnn/image35.png)

VGG-16을 활용하니 mAP가 올랐고, 데이터를 더 많이 사용하니 mAP가 더 올랐습니다.

![image.png](/assets/images/faster_rcnn/image36.png)

VGG 모델은 ZF모델 보다 복잡하기 때문에 성능은 좋은 반면 5fps로 속도는 느리다.

<**Sensitive to Hyper-parameter>**

- Hyper parameter
    - 데이터로부터 직접 학습되지 않고, 사람이 미리 설정하는 파라미터
        - 학습률
        - 배치 크기
        - 에폭
        - 네트워크 구조( 레이어 수, 유닛 수, 필터 수)
        - Faster R cnn에서의 앵커 크기, 앵커 비율, NMS 임계값

![image.png](/assets/images/faster_rcnn/image37.png)

>> 3스케일 3 비율 일때 mAP가 제일 좋습니다.

![image.png](/assets/images/faster_rcnn/image38.png)

- 앞서 말했듯 λ=10이 기본값(default)
- λ=10일 때, (정규화 후) 분류 손실과 회귀 손실 비중이 거의 비슷해지기 때문.
- λ 값이 기본값의 1/10배(λ=1)되든 10배(λ=100)되든 mAP 차이는 1% 미만입니다.
- 따라서 λ 값은 상대적으로 덜 민감한 하이퍼파라미터입니다.

**<Analysis of Recall-to-IoU>**

![image.png](/assets/images/faster_rcnn/image39.png)

• SS (Selective Search, 점선): 전통적인 Selective Search 방법으로 만든 후보영역

• EB (EdgeBoxes, 초록색 점선): 엣지 및 경계 정보를 기반으로 후보영역을 만든 EdgeBoxes 기법

• RPN ZF (파란색 실선): Region Proposal Network(RPN)을 ZF(Zeiler & Fergus) 백본으로 학습하여 만든 후보영역

• RPN VGG (빨간색 실선): RPN에 VGG 백본을 사용하여 만든 후보영역

- Recall: 실제 물체(ground-truth) 중에서, 후보영역이 일정 이상의 IoU를 만족하며 제대로 커버한 비율(검출된 비율)입니다.
- Recall = (올바르게 검출된 물체의 수) / (전체 물체의 수)
    - y축이 1에 가까울수록, "주어진 IoU 기준에서 놓친(검출되지 못한) 물체가 거의 없다"는 뜻입니다.

1) 후보영역 수가 300 → 1000 → 2000으로 늘어날수록, 모든 방법이 전반적으로 Recall이 높아집니다. (놓치는 대상이 줄어든다는 뜻)

2) RPN VGG(빨간색)는 가장 안정적으로 높은 Recall을 기록하며, RPN ZF(파란색)가 그 다음 수준을 보여줍니다.

3) Selective Search(SS)는 IoU가 조금 낮은(0.5~0.6대) 구간에서는 의외로 Recall이 꽤 높지만, IoU 요구치가 올라갈수록 Recall이 가파르게 떨어집니다. EdgeBoxes(EB)도 비슷하나 SS와 비교하면 중간 단계(0.6~0.7 주변)에서 Recall이 다소 높게 유지되다가 이후 급격히 내려갑니다.

### Conclusion

본 논문에서는 빠르고 정확한 영역 추정을 하기 위해 RPN을 제안했습니다. RPN과 객체 탐지기가 합성곱 피처를 공유함으로써 영역 추정 비용을 크게 줄였습니다. Faster R-CNN은 통합된 딥러닝 기반 객체 탐지 시스템으로, 실시간 객체 탐지가 가능할 정도로 빠릅니다. 전체적인 정확도도 기존 모델들보다 뛰어납니다.