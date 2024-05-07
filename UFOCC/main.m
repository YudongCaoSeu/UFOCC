%%
floderpath='G:\phm-ieee-2012-data-challenge-dataset\Learning_set\';               
setname='Bearing3_1';
readpath=[floderpath,setname,'\'];
fileFolder=fullfile(readpath);
dirOutput=dir(fullfile(fileFolder,'*.csv'));
fileNames={dirOutput.name};
num=length(fileNames);  
fs=25600;
% 获取传感器加速度数据
horiz_signals=zeros(num,2560);                          %%水平信号
vert_signals=zeros(num,2560);                           %%垂直信号
for i=1:num
    pp=strcat(readpath,fileNames(i));
    data = readmatrix(pp{1,1});
    horiz_signals(i,:)=data(:,5);
    vert_signals(i,:)=data(:,6);
end
%%
% x=zeros(~,~);
figure()
for i=1:size(horiz_signals,1)
    for j=1:2560
        y(2560*(i-1)+j,1)=horiz_signals(i,j);
    end
end
plot(y)
xlabel('Samples')
ylabel('Amplitude')
%set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w')
title('Horizonal vibration signal of Bearing 1-1')

%%
%A=train_score,B=test_score
figure
cri=mean(A)+3*std(A);
C=[A;B];
line([0,length(C)],[cri, cri])
hold on
plot(C)

%%
for i= 1:length(C)
  
if C(i)>=cri
   D(i)=1;
else
   D(i)=-1; 
end
end
for i= 5:length(D)
if sum(D(i-4:i))>=1
    D(i+1:length(D))=1;
end
end
figure
stem(D)
%%
seq_len=10;
D=[zeros(1,seq_len-1),D];%补充因活动窗口导致的序列缩短
figure
stem(D)
%%
%plot
FFOT=1164;
figure
subplot(2,1,1)
test_len=zeros(floor(size(Channel3,1)./10),1);
for i=1:length(test_len)
    D(i)=0;
end
hold on
%%%%Train
x_fill = [0, length(test_len), length(test_len), 0];
y_fill = [-1.2, -1.2, 1.2, 1.2];
h_train=patch(x_fill, y_fill,'r');
set(h_train, 'FaceColor', [0.87,0.91,0.77]);
%%%%Normal
x_fill = [length(test_len), FFOT, FFOT, length(test_len)];
y_fill = [-1.2, -1.2, 1.2, 1.2];
h_normal=patch(x_fill, y_fill,'r');
set(h_normal, 'FaceColor', [0.78,0.87,0.95]);
%%%%False
x_fill = [FFOT, length(D), length(D), FFOT];
y_fill = [-1.2, -1.2, 1.2, 1.2];
h_false=patch(x_fill, y_fill,'r');
set(h_false, 'FaceColor', [0.94,0.84,0.78]);
clear i
for i=1:length(D)
    if D(i)==0
        stem(i,D(i),'Color','[1,0,0]')
    elseif D(i)==-1
        stem(i,D(i),'Color','[1,0,0]')
    else
        stem(i,D(i),'Color','[1,0,0]')
    end
end
xlim([0 length(D)])
xlabel('Times(10s)/File_{num}','FontName','Times New Roman')
ylabel('Judgement','FontName','Times New Roman')
title('Bearing 1-3 [Seq_len=50, Batch_size=32]','FontName','Times New Roman','FontWeight', 'bold','Interpreter','none')
legend('Train Period','Normal Period','Fault Period','FontName','Times New Roman','FontWeight', 'bold')
ax = gca;
set(ax, 'FontName', 'Times New Roman');
hold off

subplot(2,1,2)
plot(x1,'Color','0.72,0.27,1.00')
xlabel({'Samples' '(a)'},'FontName','Times New Roman')
ylabel('Amplitude','FontName','Times New Roman')
%set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w')
title('Vibration signal','FontName','Times New Roman','FontWeight', 'bold')
ax = gca;
set(ax, 'FontName', 'Times New Roman');
xlim([0 length(x1)])

%%
% a=x+y;b=(x-y)^2;

a=linspace(0, 5, 100);
b=linspace(0, 5, 100);

% for i = 1:length(a)
%     for j = 1:length(b)
% S(i,j)=0.5.*(exp(-b(j))).*a(i)+0.5.*b(j);
%     end
% end
[X, Y] = meshgrid(a,b);
S=0.5.*(exp(-Y)).*X+0.5.*Y;
contourf(X,Y,S,10)
colormap(gca,parula);
colorbar; % 显示色条
my_handle=colorbar;
my_handle.Label.String = 'Loss Value';
xlabel('$d(\mathbf{g})+\tilde{d}(\mathbf{g})$','interpreter','latex')
ylabel('$(d(\mathbf{g})-\tilde{d}(\mathbf{g}))^2$','interpreter','latex')