# Elliptic Envelope 모델링

1. elliptic envelope 단일모델 (ipynb, 제출 csv 파일 깃헙 업로드)

→ validation macro f1 score: `0.9236496787663914`

→ test f1 score: 0.9276907062 (public), 0.9094856536 (private)

1. standard scaling (ipynb, 제출 csv 파일 깃헙 업로드)

→ validation macro f1 score: `0.9165787375726882`

→ test f1 score: 0.9284014343 (public), 0.9019219415 (private)

1. minmax scaling (ipynb, 제출 csv 파일 깃헙 업로드)

→ validation macro f1 score: `0.0014048895340425516`

→ test f1 score:

1. t-sne (ipynb, 제출 csv 파일 깃헙 업로드)

→ validation macro f1 score: `0.40662591225274825`

→ test f1 score:

1. standard scaling & t-sne (ipynb, 제출 csv 파일 깃헙 업로드)

→ validation macro f1 score: `0.4316131365747393`

→ test f1 score:

1. min max scaling & t-sne (ipynb, 제출 csv 파일 깃헙 업로드)

→ validation macro f1 score: `0.0010529271374420891`

→ test f1 score:

- 단일모델, standard scaling 적용이 높은 성능을 보임, min max scaling을 잘못한 것인지 낮은 성능을 보임, t-sne 또한 마찬가지, 향후 test score를 확인하고 문제점 파악 후 개선할 필요가 있음

## 하이퍼파라미터 - randomizedsearchcv 이용 (iteration = 20, cv = 5)

- sklearn.covariance.EllipticEnvelope 공식문서
  - ![스크린샷 2023-02-17 오후 10.11.23](/Users/jeonghwan/Desktop/스크린샷 2023-02-17 오후 10.11.23.png)

* contamination: EllipticEnvelope의 이상치 비율을 설정하는 매개변수, 예를 들어 'contamination'이 0.01이면 데이터의 1%가 이상치로 간주된다.

- support_fraction: 분산 추정에 사용되는 포인트의 비율을 설정하는 매개변수, EllipticEnvelope은 이상치를 분류하기 위해 포인트 간의 분산을 추정하는데, 'support_fraction' 값이 작을수록 추정된 분산은 더 큰 영역을 포함하게 되며, 이는 더 많은 포인트를 이상치로 분류할 가능성이 높아지는 것이다.

- 'contamination': 0.0004745401188473625, 'support_fraction': 0.9995071430640992

![스크린샷 2023-02-17 오후 8.19.51](/Users/jeonghwan/Library/Application Support/typora-user-images/스크린샷 2023-02-17 오후 8.19.51.png)

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.covariance import EllipticEnvelope
from scipy.stats import uniform

param_dist = {
    'support_fraction': uniform(0.99, 0.01),
    'contamination': uniform(0.0001, 0.001),
}

model = EllipticEnvelope(random_state=42)
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=5, random_state=42, verbose=2)

random_search.fit(train_x)

print(f'Best parameters: {random_search.best_params_}')
print(f'Best score: {random_search.best_score_:.4f}')
```

![스크린샷 2023-02-17 오후 8.20.10](/Users/jeonghwan/Library/Application Support/typora-user-images/스크린샷 2023-02-17 오후 8.20.10.png)

- validation F1 score: 0.7887218676684034

- 1시간 정도 걸렸는데, 하이퍼파라미터마다 탐색범위를 더 넓히거나 grid search를 이용하면 성능을 더 높일 수 있지만 몇 시간으로 안될 수도 있음

- elliptic envelope을 활용한 수상작 코드 확인 후 높은 성능을 보인 하이퍼파라미터를 가져와서 사용해 성능을 높였음 (

  ```
  0.9236496787663914
  ```

  )

  - elliptic envelope을 최종 모델로 선정한다면 시간을 들여서라도 범위를 넓힌 grid search를 이용해 최적 하이퍼파라미터를 찾고 성능을 향상시킬 수 있을 것

