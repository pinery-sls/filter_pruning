# Filter Pruning via Measuring Feature Map Information

Dataset： CIFAR10/100  ImageNet

CIFAR: VGG ResNet DenseNet
ImageNet: ResNet

## Result of CIFAR10

\begin{table}[b]
	\caption{Pruning reasult on CIFAR10}
	\centering
	\begin{tabular}{c|cccc}
		\hline Model &  Alg &  Acc(\%) & Param & FLOPs  \\
		\hline \multirow{9}*{ VGG16 } & Baseline  & 93.90 & 14.72 M & 313.75 M \\
		& NS\citep{liu2017learning} & 93.69 & 3.45 M & 199.66 M \\
		& L1\citep{li2016pruning} & 93.40 & 5.40 M & 206.00 M \\
		& SSS\citep{huang2018data} & 93.02 & 3.93 M & 183.13M \\
		& GAL-0.05\citep{lin2019towards} & 92.03 & 3.36 M & 189.49 M \\
		& VCNNP\citep{zhao2019variational} & 93.18 & 3.92 M & 190.01 M \\
		& HRank\citep{lin2020hrank} & 92.34 & 2.64 M & 108.61 M \\
		& CPMC\citep{yan2020channel} & 93.40 & 1.04 M & 106.68 M \\
		& $\mathbf{Our-E}$ & $\mathbf{93.53}$ & $\mathbf{0.99 M}$ & $\mathbf{83.96 M}$ \\
		& $\mathbf{Our-P}$ & 93.47 & $\mathbf{0.93 M}$ & $\mathbf{89.02 M}$ \\
		& $\mathbf{Our-P}$ & 93.16 & $\mathbf{0.90 M}$ & $\mathbf{79.85 M}$ \\
		\hline \multirow{3}*{ VGG19 } &  Baseline & 93.68 & 20.04 M & 398.74 M \\
		& NS\citep{liu2017learning} & 93.66 & 2.43 M & 208.54 M \\
		& $\mathbf{Our-E}$ & 93.63 & $\mathbf{1.55M}$ & 129.21 M \\
		& $\mathbf{Our-P}$ & 93.58 & $\mathbf{1.45M}$ & 127.44 M \\
		\hline \multirow{9}*{ ResNet56 }  & Baseline & 93.22 & 0.85 M & 126.55 M \\
		& NS\citep{liu2017learning} & 92.94 & 0.41 M & 64.94 M \\
		& L1\citep{li2016pruning} & 93.06 & 0.73 M & 90.90 M \\
		& NISP\citep{yu2018nisp} & 93.01 & 0.49 M & 81.00 M \\
		& GAL-0.6\citep{lin2019towards} & 92.98 & 0.75 M & 78.30 M \\
		& HRank\citep{lin2020hrank} & 93.17 & 0.49 M & 62.72 M \\
		& KSE (G=4)\citep{li2019exploiting} & 93.23 & 0.43 M & 60 M \\
		& KSE (G=5)\citep{li2019exploiting} & 92.88 & 0.36 M & 50 M \\
		& $\mathbf{Our-E}$ & $\mathbf{93.56}$ & 0.39 M & 69.52 M \\
		& $\mathbf{Our-P}$ & 93.36 & 0.39 M & 63.15 M \\
		& $\mathbf{Our-P}$ & 93.09 & $\mathbf{0.31 M}$ & 59.66 M \\
		\hline \multirow{4}*{ ResNet164 }  & Baseline & 95.04 & 1.71 M & 254.50 M \\
		& NS\citep{liu2017learning} & 94.73 & 1.10 M & 137.50 M\\
		& CPMC\citep{yan2020channel} & 94.76 & 0.75 M & 144.02 M \\
		& $\mathbf{Our-E}$ & 94.66 & $\mathbf{0.67 M}$ & $\mathbf{111.33 M}$ \\
		& $\mathbf{Our-P}$ & 93.65 & 0.73 M & 105.86 M \\
		\hline \multirow{8}*{ DenseNet40 } & Baseline & 94.26 & 1.06 M & 290.13 M \\
		& GAL-0.01\citep{lin2019towards} & 94.9 & 0.67 M & 182.92 M \\
		& HRank\citep{lin2020hrank} & 94.24 & 0.66 M & 167.41 M \\
		& VCNNP\citep{zhao2019variational} & 93.16 & 0.42 M & 156.00 M \\
		& CPMC\citep{yan2020channel} & 93.74 & 0.42 M & 121.73 M \\
		& KSE (G=6)\citep{li2019exploiting} & 94.70 & 0.39 M & 115 M \\
		& NS\citep{liu2017learning} & 94.09 & 0.40 M & 132.16 M \\
		& $\mathbf{Our-E}$ & 94.04 & $\mathbf{0.38 M}$ & $\mathbf{110.72 M}$ \\
		& $\mathbf{Our-P}$ & 93.75 & $\mathbf{0.37 M}$ & $\mathbf{100.12 M}$ \\
		\hline
	\end{tabular}
	\label{cifar10}
\end{table}

## Result of CIFAR100
\begin{table}[b]
	\caption{Pruning result on CIFAR100}
	\centering
	\begin{tabular}{c|cccc}
		\hline  Model & Alg & Acc(\%) & Param & FLOPs \\
		\hline \multirow{6}{*}{ VGG16 } & Baseline & 73.80 & 14.77 M & 313.8 M \\
		& VCNNP\citep{zhao2019variational} & 73.33 & 9.14 M & 256.00 M \\
		& NS\citep{liu2017learning} & 73.72 & 8.83 M & 274.00 M \\
		& CPGMI\citep{lee2020channel} & 73.53 & 4.99 M & 198.20 M \\
		& CPMC\citep{yan2020channel} & 73.01 & 4.80 M & 162.00 M \\
		& $\mathbf{Ours-E}$ & 73.17 & $\mathbf{4.94 M}$ & $\mathbf{150.70 M}$ \\
		& $\mathbf{Ours-E}$ & 73.06 & $\mathbf{4.05 M}$ & $\mathbf{129.52 M}$ \\
		& $\mathbf{Ours-P}$ & 73.17 & 4.09 M & 147.99 M \\
		\hline \multirow{3}{*}{ VGG19 } &  Baseline & 73.81 & 20.08 M & 398.79 M \\
		&  NS\citep{liu2017learning} & 73.00 & 5.84 M & 274.36 M \\
		&  $\mathbf{Ours-E}$ & $\mathbf{73.29}$ & 4.21 M & $\mathbf{183.69 M}$ \\
		&  $\mathbf{Ours-P}$ & $\mathbf{73.15}$ & $\mathbf{4.17 M}$ & 195.77 M \\
		&  $\mathbf{Ours-P}$ & $\mathbf{73.01}$ & $\mathbf{3.94 M}$ & 180.51 M \\
		\hline \multirow{4}{*}{ ResNet56 } & Baseline  & 71.77 & 0.86 M & 71.77 M \\
		& NS\citep{liu2017learning} & 70.51 & 0.60 M & 62,82 M \\
		& $\mathbf{Ours-E}$ & $\mathbf{71.28}$ & 0.50 M & 80.48 M \\
		& $\mathbf{Ours-E}$ & 70.67 & $\mathbf{0.41 M}$ & 69.88 M \\
		\hline \multirow{4}{*}{ ResNet164 } & Baseline & 76.74 & 1.73 M & 253.97 M \\
		& NS\citep{liu2017learning} & 76.18 & 1.21 M & 123.50 M \\
		& CPMC\citep{yan2020channel} & 77.22 & 0.96 M & 151.92 M \\
		& $\mathbf{Ours-E}$ & 76.28 & $\mathbf{0.94 M}$ & 150.57 M \\
		& $\mathbf{Ours-P}$ & 75.27 & $\mathbf{0.94 M}$ & $\mathbf{123.09 M}$ \\
		\hline \multirow{6}{*}{ DenseNet40 } & Baseline & 74.37 & 1.11 M & 287.75 M \\
		& VCNNP\citep{zhao2019variational} & 72.19 & 0.65 M & 218.00 M \\
		& CPGMI\citep{lee2020channel} & 73.84 & 0.66 M & 198.50 M \\
		& CPMC\citep{yan2020channel} & 73.93 & 0.58 M & 155.24 M \\
		& NS\citep{liu2017learning} & 73.87 & 0.55 M & 164.36 M \\
		& $\mathbf{Ours-E}$  & $\mathbf{74.50}$ & $\mathbf{0.40M}$ & $\mathbf{109.55 M}$ \\
		& $\mathbf{Ours-E}$  & 73.74 & $\mathbf{0.34M}$ & $\mathbf{95.79 M}$ \\
		& $\mathbf{Our-P}$ & $\mathbf{74.26}$ & $\mathbf{0.39M}$ & $\mathbf{108.81M}$ \\
		& $\mathbf{Our-P}$ & 73.62 & $\mathbf{0.34M}$ & $\mathbf{94.84M}$ \\
		\hline
	\end{tabular}
	\label{cifar100}
\end{table}

## Result of ImageNet
\begin{table}[ht]
	\caption{Pruning results of ResNet50 on ImageNet.}
	\centering
	\begin{tabular}{ccccc}
		\hline Model & Top-1\% & Top-5\% &  FLOPs & Parameters \\
		\hline ResNet50\citep{luo2017thinet} & 76.15 & 92.87 & 4.09 $\mathrm{~B}$ & 25.50 $\mathrm{M}$ \\
		SSS-32\citep{huang2018data}  & 74.18 & 91.91 & 2.82 $\mathrm{~B}$ & 18.60 $\mathrm{M}$ \\
		\citep{he2017channel} & 72.30 & 90.80 & 2.73 $\mathrm{~B}$ & - \\
		GAL-0.5\citep{lin2019towards} & 71.95 & 90.94 & 2.33 $\mathrm{~B}$ & 21.20 $\mathrm{M}$ \\
		HRank\citep{lin2020hrank}  & 74.98 & 92.33 & 2.30 $\mathrm{~B}$ & 16.15 $\mathrm{M}$ \\
		GDP-0.6\citep{lin2018accelerating} & 71.19 & 90.71 & 1.88 $\mathrm{~B}$ & - \\
		GDP-0.5\citep{lin2018accelerating} & 69.58 & 90.14 & 1.57 $\mathrm{~B}$ & - \\
		SSS-26\citep{huang2018data}  & 71.82 & 90.79 & 2.33 $\mathrm{~B}$ & 15.60 $\mathrm{M}$ \\
		GAL-1\citep{lin2019towards}  & 69.88 & 89.75 & 1.58 $\mathrm{~B}$ & 14.67 $\mathrm{M}$ \\
		GAL-0.5-joint\citep{lin2019towards}  & 71.80 & 90.82 & 1.84 $\mathrm{~B}$ & 19.31 $\mathrm{M}$ \\
		HRank\citep{lin2020hrank} & 71.98 & 91.01 & 1.55 $\mathrm{~B}$ & 13.77 $\mathrm{M}$ \\
		ThiNet-50\citep{luo2017thinet}  & 68.42 & 88.30 & 1.10 $\mathrm{~B}$ & 8.66 $\mathrm{M}$ \\
		GAL-1-joint\citep{lin2019towards}  & 69.31 & 89.12 & 1.11 $\mathrm{~B}$ & 10.21 $\mathrm{M}$ \\
		HRank\citep{lin2020hrank} & 69.10 & 89.58 & 0.98 $\mathrm{~B}$ & 8.27 $\mathrm{M}$ \\
		NS\citep{liu2017learning} & 70.43 & 89.93 &  $\mathrm{2.54 B}$ &  $\mathrm{18.33 M}$ \\
		$\mathbf{Ours-E}$ & $\mathbf{72.02}$ & 90.69 &  $\mathrm{1.84 B}$ &  $\mathrm{11.41 M}$ \\
		$\mathbf{Ours-E}$ & 70.41 & 89.91 &  $\mathrm{1.41 B}$ &  $\mathrm{8.51 M}$ \\
		$\mathbf{Ours-P}$ & 69.91 & 89.46 &  $\mathrm{1.70 B}$ &  $\mathrm{11.06 M}$ \\
		$\mathbf{Ours-P}$ & 68.62 & 88.62 &  $\mathrm{1.34 B}$ &  $\mathrm{8.23 M}$ \\
		\hline
	\end{tabular}
	\label{imagenet}
\end{table}

