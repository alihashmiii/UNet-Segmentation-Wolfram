(* ::Package:: *)

BeginPackage["UNETSegmentation`"]


(* ::Section:: *)
(*Creating  UNet*)


conv[n_]:=NetChain[
{
 ConvolutionLayer[n,3,"PaddingSize"->{1,1}],
 Ramp,
 BatchNormalizationLayer[],
 ConvolutionLayer[n,3,"PaddingSize"->{1,1}],
 Ramp,
 BatchNormalizationLayer[]
}
];


pool := PoolingLayer[{2,2},2];


dec[n_]:=NetGraph[
{
 "deconv" -> DeconvolutionLayer[n,{2,2},"Stride"->{2,2}],
 "cat" -> CatenateLayer[],
 "conv" -> conv[n]
},
{
 NetPort["Input1"]->"cat",
 NetPort["Input2"]->"deconv"->"cat"->"conv"
}
];


nodeGraphMXNET[net_,opt: ("MXNetNodeGraph"|"MXNetNodeGraphPlot")]:= net~NetInformation~opt;


UNET := NetGraph[
<|
"enc_1"-> conv[64],
"enc_2"-> {pool,conv[128]},
"enc_3"-> {pool,conv[256]},
"enc_4"-> {pool,conv[512]},
"enc_5"-> {pool,conv[1024]},
"dec_1"-> dec[512],
"dec_2"-> dec[256],
"dec_3"-> dec[128],
"dec_4"-> dec[64],
"map"->{ConvolutionLayer[1,{1,1}],LogisticSigmoid}
|>,
{
NetPort["Input"]->"enc_1"->"enc_2"->"enc_3"->"enc_4"->"enc_5",
{"enc_4","enc_5"}->"dec_1",
{"enc_3","dec_1"}->"dec_2",
{"enc_2","dec_2"}->"dec_3",
{"enc_1","dec_3"}->"dec_4",
"dec_4"->"map"},
"Input"->NetEncoder[{"Image",{160,160},ColorSpace->"Grayscale"}]
]


(* ::Section:: *)
(*DataPrep*)


dataPrep[dirImage_,dirMask_]:=Module[{X, masks,imgfilenames, maskfilenames,ordering, fNames},
SetDirectory[dirImage];
fNames = FileNames[];
ordering=Flatten@StringCases[fNames,x_~~p:DigitCharacter..:>ToExpression@p];
imgfilenames = Part[fNames,Ordering@ordering];
X = ImageResize[Import[dirImage<>"\\"<>#],{160,160}]&/@imgfilenames;
SetDirectory[dirMask];
fNames = FileNames[];
ordering=Flatten@StringCases[fNames,x_~~p:DigitCharacter..:>ToExpression@p];
maskfilenames = Part[fNames,Ordering@ordering];
masks = Import[dirMask<>"\\"<>#]&/@maskfilenames;
{X, NetEncoder[{"Image",{160,160},ColorSpace->"Grayscale"}]/@masks}
]


(* ::Section:: *)
(*Training UNet*)


trainNetwithValidation[net_,dataset_,labeldataset_,validationset_,labelvalidationset_, batchsize_: 8, maxtrainRounds_: 100]:=Module[{},
 SetDirectory[NotebookDirectory[]];
 NetTrain[net, dataset->labeldataset,All, ValidationSet -> Thread[validationset-> labelvalidationset],
 BatchSize->batchsize,MaxTrainingRounds->maxtrainRounds, TargetDevice->"GPU",
 TrainingProgressCheckpointing->{"Directory","results","Interval"->Quantity[5,"Rounds"]}]
];


trainNet[net_,dataset_,labeldataset_, batchsize_:8, maxtrainRounds_: 10]:=Module[{},
 SetDirectory[NotebookDirectory[]];
 NetTrain[net, dataset->labeldataset,All,BatchSize->batchsize,MaxTrainingRounds->maxtrainRounds, TargetDevice->"GPU",
 TrainingProgressCheckpointing->{"Directory","results","Interval"-> Quantity[5,"Rounds"]}]
];


(* ::Section:: *)
(*Measure Accuracy*)


measureModelAccuracy[net_,data_,groundTruth_]:= Module[{acc},
acc =Table[{i, 1.0 - HammingDistance[N@Round@Flatten@net[data[[i]],TargetDevice->"GPU"],
 Flatten@groundTruth[[i]]]/(160*160)},{i,Length@data}
];
{Mean@Part[acc,All,2],TableForm@acc}
];


(* ::Section:: *)
(*Miscellaneous*)


saveNeuralNet[net_]:= Module[{dir = NotebookDirectory[]},
 Export[dir<>"unet.wlnet",net]
]/; Head[net]=== NetGraph;


saveInputs[data_,labels_,opt:("data"|"validation")]:=Module[{},
 SetDirectory[NotebookDirectory[]];
 Switch[opt,"data",
  Export["X.mx",data];Export["Y.mx",labels],
  "validation",
  Export["Xval.mx",data];Export["Yval.mx",labels]
 ]
]


EndPackage[];
