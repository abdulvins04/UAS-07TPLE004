{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline \n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import load_digits\n",
    "from sklearn.model_selection import train_test_split\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn import datasets\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "from sklearn.decomposition import PCA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "digits = load_digits()\n",
    "iris = datasets.load_iris()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['sepal length (cm)',\n",
       " 'sepal width (cm)',\n",
       " 'petal length (cm)',\n",
       " 'petal width (cm)']"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris.feature_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1797, 64)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "digits.data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.,  0.,  1.,  9., 15., 11.,  0.,  0.,  0.,  0., 11., 16.,  8.,\n",
       "       14.,  6.,  0.,  0.,  2., 16., 10.,  0.,  9.,  9.,  0.,  0.,  1.,\n",
       "       16.,  4.,  0.,  8.,  8.,  0.,  0.,  4., 16.,  4.,  0.,  8.,  8.,\n",
       "        0.,  0.,  1., 16.,  5.,  1., 11.,  3.,  0.,  0.,  0., 12., 12.,\n",
       "       10., 10.,  0.,  0.,  0.,  0.,  1., 10., 13.,  3.,  0.,  0.])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "digits.data[10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "digits.target_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1797,)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "digits.target.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "digits.target[10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x13b0ec30>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 432x288 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt \n",
    "plt.gray() \n",
    "plt.matshow(digits.images[10]) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = digits.data\n",
    "y = digits.target\n",
    "# Membagi data training dan testing(80:20)\n",
    "x_train, x_test, y_train, y_test = train_test_split(X,y, test_size= 0.4, random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1078, 64)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(719, 64)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1078,)"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(719,)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "clf = MLPClassifier(hidden_layer_sizes=(20,20,20),  max_iter=250, alpha=0.0001,activation='logistic',\n",
    "                     solver='adam', verbose=10,  random_state=21,tol=0.000000001)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 1, loss = 2.36734712\n",
      "Iteration 2, loss = 2.34817172\n",
      "Iteration 3, loss = 2.33271425\n",
      "Iteration 4, loss = 2.32050890\n",
      "Iteration 5, loss = 2.31091769\n",
      "Iteration 6, loss = 2.30406517\n",
      "Iteration 7, loss = 2.29836173\n",
      "Iteration 8, loss = 2.29344395\n",
      "Iteration 9, loss = 2.28980259\n",
      "Iteration 10, loss = 2.28627223\n",
      "Iteration 11, loss = 2.28263125\n",
      "Iteration 12, loss = 2.27876695\n",
      "Iteration 13, loss = 2.27498134\n",
      "Iteration 14, loss = 2.27064775\n",
      "Iteration 15, loss = 2.26576479\n",
      "Iteration 16, loss = 2.26036642\n",
      "Iteration 17, loss = 2.25430620\n",
      "Iteration 18, loss = 2.24754610\n",
      "Iteration 19, loss = 2.23979390\n",
      "Iteration 20, loss = 2.23100170\n",
      "Iteration 21, loss = 2.22132593\n",
      "Iteration 22, loss = 2.21032674\n",
      "Iteration 23, loss = 2.19838079\n",
      "Iteration 24, loss = 2.18464374\n",
      "Iteration 25, loss = 2.16941498\n",
      "Iteration 26, loss = 2.15348046\n",
      "Iteration 27, loss = 2.13547414\n",
      "Iteration 28, loss = 2.11577265\n",
      "Iteration 29, loss = 2.09502834\n",
      "Iteration 30, loss = 2.07329350\n",
      "Iteration 31, loss = 2.05035541\n",
      "Iteration 32, loss = 2.02699971\n",
      "Iteration 33, loss = 2.00318848\n",
      "Iteration 34, loss = 1.97891969\n",
      "Iteration 35, loss = 1.95499812\n",
      "Iteration 36, loss = 1.93096305\n",
      "Iteration 37, loss = 1.90739017\n",
      "Iteration 38, loss = 1.88393448\n",
      "Iteration 39, loss = 1.86142610\n",
      "Iteration 40, loss = 1.83907261\n",
      "Iteration 41, loss = 1.81747214\n",
      "Iteration 42, loss = 1.79764459\n",
      "Iteration 43, loss = 1.77709483\n",
      "Iteration 44, loss = 1.75708265\n",
      "Iteration 45, loss = 1.74017474\n",
      "Iteration 46, loss = 1.72084895\n",
      "Iteration 47, loss = 1.70378977\n",
      "Iteration 48, loss = 1.68622787\n",
      "Iteration 49, loss = 1.66882117\n",
      "Iteration 50, loss = 1.65212882\n",
      "Iteration 51, loss = 1.63550412\n",
      "Iteration 52, loss = 1.61818743\n",
      "Iteration 53, loss = 1.60208315\n",
      "Iteration 54, loss = 1.58485072\n",
      "Iteration 55, loss = 1.56861866\n",
      "Iteration 56, loss = 1.55132498\n",
      "Iteration 57, loss = 1.53499339\n",
      "Iteration 58, loss = 1.51834979\n",
      "Iteration 59, loss = 1.50096532\n",
      "Iteration 60, loss = 1.48481352\n",
      "Iteration 61, loss = 1.46913654\n",
      "Iteration 62, loss = 1.45199801\n",
      "Iteration 63, loss = 1.43507232\n",
      "Iteration 64, loss = 1.41923035\n",
      "Iteration 65, loss = 1.40349466\n",
      "Iteration 66, loss = 1.38784639\n",
      "Iteration 67, loss = 1.37288427\n",
      "Iteration 68, loss = 1.35732686\n",
      "Iteration 69, loss = 1.34373374\n",
      "Iteration 70, loss = 1.32852261\n",
      "Iteration 71, loss = 1.31507641\n",
      "Iteration 72, loss = 1.30132733\n",
      "Iteration 73, loss = 1.28788633\n",
      "Iteration 74, loss = 1.27450751\n",
      "Iteration 75, loss = 1.26202316\n",
      "Iteration 76, loss = 1.25014068\n",
      "Iteration 77, loss = 1.23813579\n",
      "Iteration 78, loss = 1.22638735\n",
      "Iteration 79, loss = 1.21420797\n",
      "Iteration 80, loss = 1.20268819\n",
      "Iteration 81, loss = 1.19184658\n",
      "Iteration 82, loss = 1.18136723\n",
      "Iteration 83, loss = 1.16944204\n",
      "Iteration 84, loss = 1.15826283\n",
      "Iteration 85, loss = 1.14766899\n",
      "Iteration 86, loss = 1.13677378\n",
      "Iteration 87, loss = 1.12601891\n",
      "Iteration 88, loss = 1.11594658\n",
      "Iteration 89, loss = 1.10597157\n",
      "Iteration 90, loss = 1.09673901\n",
      "Iteration 91, loss = 1.08659875\n",
      "Iteration 92, loss = 1.07733869\n",
      "Iteration 93, loss = 1.06820385\n",
      "Iteration 94, loss = 1.05911235\n",
      "Iteration 95, loss = 1.05001466\n",
      "Iteration 96, loss = 1.04095850\n",
      "Iteration 97, loss = 1.03243896\n",
      "Iteration 98, loss = 1.02371691\n",
      "Iteration 99, loss = 1.01491994\n",
      "Iteration 100, loss = 1.00641962\n",
      "Iteration 101, loss = 0.99753641\n",
      "Iteration 102, loss = 0.98937448\n",
      "Iteration 103, loss = 0.98082259\n",
      "Iteration 104, loss = 0.97234001\n",
      "Iteration 105, loss = 0.96452374\n",
      "Iteration 106, loss = 0.95619655\n",
      "Iteration 107, loss = 0.94749471\n",
      "Iteration 108, loss = 0.93946981\n",
      "Iteration 109, loss = 0.93171869\n",
      "Iteration 110, loss = 0.92352460\n",
      "Iteration 111, loss = 0.91568614\n",
      "Iteration 112, loss = 0.90781969\n",
      "Iteration 113, loss = 0.90045647\n",
      "Iteration 114, loss = 0.89265188\n",
      "Iteration 115, loss = 0.88534228\n",
      "Iteration 116, loss = 0.87764479\n",
      "Iteration 117, loss = 0.86973801\n",
      "Iteration 118, loss = 0.86265081\n",
      "Iteration 119, loss = 0.85541627\n",
      "Iteration 120, loss = 0.84780316\n",
      "Iteration 121, loss = 0.84079732\n",
      "Iteration 122, loss = 0.83369450\n",
      "Iteration 123, loss = 0.82653739\n",
      "Iteration 124, loss = 0.81955655\n",
      "Iteration 125, loss = 0.81255512\n",
      "Iteration 126, loss = 0.80592202\n",
      "Iteration 127, loss = 0.79982742\n",
      "Iteration 128, loss = 0.79288619\n",
      "Iteration 129, loss = 0.78615932\n",
      "Iteration 130, loss = 0.77947845\n",
      "Iteration 131, loss = 0.77341180\n",
      "Iteration 132, loss = 0.76703008\n",
      "Iteration 133, loss = 0.76040541\n",
      "Iteration 134, loss = 0.75435339\n",
      "Iteration 135, loss = 0.74853416\n",
      "Iteration 136, loss = 0.74168078\n",
      "Iteration 137, loss = 0.73612334\n",
      "Iteration 138, loss = 0.73041563\n",
      "Iteration 139, loss = 0.72421019\n",
      "Iteration 140, loss = 0.71827922\n",
      "Iteration 141, loss = 0.71248624\n",
      "Iteration 142, loss = 0.70676180\n",
      "Iteration 143, loss = 0.70107109\n",
      "Iteration 144, loss = 0.69519668\n",
      "Iteration 145, loss = 0.68962563\n",
      "Iteration 146, loss = 0.68441358\n",
      "Iteration 147, loss = 0.67900416\n",
      "Iteration 148, loss = 0.67331430\n",
      "Iteration 149, loss = 0.66813341\n",
      "Iteration 150, loss = 0.66302262\n",
      "Iteration 151, loss = 0.65750735\n",
      "Iteration 152, loss = 0.65225149\n",
      "Iteration 153, loss = 0.64750873\n",
      "Iteration 154, loss = 0.64192460\n",
      "Iteration 155, loss = 0.63718734\n",
      "Iteration 156, loss = 0.63226594\n",
      "Iteration 157, loss = 0.62746514\n",
      "Iteration 158, loss = 0.62255851\n",
      "Iteration 159, loss = 0.61780144\n",
      "Iteration 160, loss = 0.61352194\n",
      "Iteration 161, loss = 0.60845183\n",
      "Iteration 162, loss = 0.60431094\n",
      "Iteration 163, loss = 0.59940481\n",
      "Iteration 164, loss = 0.59469051\n",
      "Iteration 165, loss = 0.59025338\n",
      "Iteration 166, loss = 0.58631529\n",
      "Iteration 167, loss = 0.58136250\n",
      "Iteration 168, loss = 0.57690825\n",
      "Iteration 169, loss = 0.57259851\n",
      "Iteration 170, loss = 0.56861541\n",
      "Iteration 171, loss = 0.56394426\n",
      "Iteration 172, loss = 0.55980897\n",
      "Iteration 173, loss = 0.55584012\n",
      "Iteration 174, loss = 0.55158150\n",
      "Iteration 175, loss = 0.54718134\n",
      "Iteration 176, loss = 0.54354159\n",
      "Iteration 177, loss = 0.53918404\n",
      "Iteration 178, loss = 0.53542071\n",
      "Iteration 179, loss = 0.53135764\n",
      "Iteration 180, loss = 0.52770663\n",
      "Iteration 181, loss = 0.52363072\n",
      "Iteration 182, loss = 0.51969984\n",
      "Iteration 183, loss = 0.51568783\n",
      "Iteration 184, loss = 0.51173440\n",
      "Iteration 185, loss = 0.50799487\n",
      "Iteration 186, loss = 0.50397132\n",
      "Iteration 187, loss = 0.50034925\n",
      "Iteration 188, loss = 0.49633166\n",
      "Iteration 189, loss = 0.49245908\n",
      "Iteration 190, loss = 0.48883655\n",
      "Iteration 191, loss = 0.48488619\n",
      "Iteration 192, loss = 0.48084663\n",
      "Iteration 193, loss = 0.47706075\n",
      "Iteration 194, loss = 0.47290694\n",
      "Iteration 195, loss = 0.46894005\n",
      "Iteration 196, loss = 0.46499053\n",
      "Iteration 197, loss = 0.46067255\n",
      "Iteration 198, loss = 0.45653702\n",
      "Iteration 199, loss = 0.45251020\n",
      "Iteration 200, loss = 0.44817936\n",
      "Iteration 201, loss = 0.44382334\n",
      "Iteration 202, loss = 0.43950179\n",
      "Iteration 203, loss = 0.43482287\n",
      "Iteration 204, loss = 0.43054633\n",
      "Iteration 205, loss = 0.42622685\n",
      "Iteration 206, loss = 0.42118999\n",
      "Iteration 207, loss = 0.41681970\n",
      "Iteration 208, loss = 0.41196636\n",
      "Iteration 209, loss = 0.40735206\n",
      "Iteration 210, loss = 0.40317218\n",
      "Iteration 211, loss = 0.39779262\n",
      "Iteration 212, loss = 0.39271876\n",
      "Iteration 213, loss = 0.38829396\n",
      "Iteration 214, loss = 0.38306914\n",
      "Iteration 215, loss = 0.37797086\n",
      "Iteration 216, loss = 0.37294365\n",
      "Iteration 217, loss = 0.36838312\n",
      "Iteration 218, loss = 0.36366202\n",
      "Iteration 219, loss = 0.35879714\n",
      "Iteration 220, loss = 0.35429422\n",
      "Iteration 221, loss = 0.34962097\n",
      "Iteration 222, loss = 0.34579051\n",
      "Iteration 223, loss = 0.34089824\n",
      "Iteration 224, loss = 0.33653801\n",
      "Iteration 225, loss = 0.33240085\n",
      "Iteration 226, loss = 0.32832740\n",
      "Iteration 227, loss = 0.32454160\n",
      "Iteration 228, loss = 0.32060355\n",
      "Iteration 229, loss = 0.31650436\n",
      "Iteration 230, loss = 0.31297660\n",
      "Iteration 231, loss = 0.30924760\n",
      "Iteration 232, loss = 0.30546965\n",
      "Iteration 233, loss = 0.30226395\n",
      "Iteration 234, loss = 0.29889334\n",
      "Iteration 235, loss = 0.29520362\n",
      "Iteration 236, loss = 0.29166868\n",
      "Iteration 237, loss = 0.28842900\n",
      "Iteration 238, loss = 0.28511433\n",
      "Iteration 239, loss = 0.28210240\n",
      "Iteration 240, loss = 0.27838691\n",
      "Iteration 241, loss = 0.27538281\n",
      "Iteration 242, loss = 0.27243601\n",
      "Iteration 243, loss = 0.26939424\n",
      "Iteration 244, loss = 0.26625555\n",
      "Iteration 245, loss = 0.26273998\n",
      "Iteration 246, loss = 0.26025299\n",
      "Iteration 247, loss = 0.25729051\n",
      "Iteration 248, loss = 0.25442468\n",
      "Iteration 249, loss = 0.25118925\n",
      "Iteration 250, loss = 0.24868404\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\users\\dapodik\\appdata\\local\\programs\\python\\python37-32\\lib\\site-packages\\sklearn\\neural_network\\multilayer_perceptron.py:562: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (250) reached and the optimization hasn't converged yet.\n",
      "  % self.max_iter, ConvergenceWarning)\n"
     ]
    }
   ],
   "source": [
    "clf.fit(x_train, y_train)\n",
    "y_pred = clf.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "loss_values = clf.loss_curve_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x13c25370>]"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(loss_values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
