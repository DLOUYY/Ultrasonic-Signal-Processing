clear all;
close all;

% '500.txt' in the code is obtained from the following Baidu network disk link
% Baidu network disk link ��https://pan.baidu.com/s/1kUEYQhHgtcGxbuUCAjKtaQ  
% Extraction code��7c23

fprintf(2,'Error using load\n');
fprintf(2,'Unable to read file 500.txt . No such file or directory.\n');
disp('If the above error message appears, please go to the Baidu network disk link to download 500.txt ');
disp('Baidu network disk link ��https://pan.baidu.com/s/1kUEYQhHgtcGxbuUCAjKtaQ  ');
disp('Extraction code��7c23 ');



fs=120; 
T=10;
t=1/fs:1/fs:T;
f1=2.*exp(-3.*abs(t-7)).*cos(2.*pi.*5.*(t-7))+exp(-6.*abs(t-2)).*cos(2.*pi.*30.*(t-2));
f = f1 + 0.1*rand(1,fs*T);


figure;
subplot(2,2,1);plot(t,f1,'k');grid on;
subplot(2,2,2);hua_fft(f1,fs,1);grid on;
 subplot(2,2,3);plot(t,f,'k');grid on;
subplot(2,2,4);hua_fft(f,fs,1);grid on;

%% ----------------------EMD------------------------

[imf,ort,nbits]=emd(f);

figure('Name', 'EMD�ֽ�');
subplot(6,2,1);plot(t,f,'k','linewidth',1.2);grid on;
subplot(6,2,2);hua_fft(f,fs,1);grid on;

for i = 2:6
    subplot(6,2,i*2-1);
    plot(t,imf(i-1,:),'k','linewidth',1.2);ylabel(['IMF',int2str(i-1)]);grid on;
    subplot(6,2,i*2);
    hua_fft(imf(i-1,:),fs,1);ylabel(['IMF',int2str(i-1)]);grid on;
end

figure;
for i = 1:size(imf,1)-5
    subplot(6,2,i*2-1);
    plot(t,imf(i+5,:),'k','linewidth',1.2);ylabel(['IMF',int2str(i+5)]);grid on;
    subplot(6,2,i*2);
    hua_fft(imf(i+5,:),fs,1);ylabel(['IMF',int2str(i+5)]);grid on;
end

corrall=corrcoef([f' imf']);
corr=corrall(1,2:end);    


%% ---------------------EEMD----------------
[imf_eemd its_eemd] = eemd(f,0.2,100,5000);

figure('Name', 'EEMD�ֽ�');
subplot(6,2,1);plot(t,f,'k','linewidth',1.2);grid on;
subplot(6,2,2);hua_fft(f,fs,1);grid on;
for i = 2:6
    subplot(6,2,i*2-1);
    plot(t,imf_eemd(i-1,:),'k','linewidth',1.2);ylabel(['IMF',int2str(i-1)]);grid on;
    subplot(6,2,i*2);
    hua_fft(imf_eemd(i-1,:),fs,1);ylabel(['IMF',int2str(i-1)]);grid on;
end

figure;
for i = 1:size(imf_eemd,1)-5
    subplot(6,2,i*2-1);
    plot(t,imf_eemd(i+5,:),'k','linewidth',1.2);ylabel(['IMF',int2str(i+5)]);grid on;
    subplot(6,2,i*2);
    hua_fft(imf_eemd(i+5,:),fs,1);ylabel(['IMF',int2str(i+5)]);grid on;
end

corrall_eemd=corrcoef([f' imf_eemd']);
corr_eemd=corrall_eemd(1,2:end);  

%% -------------------CEEMDAN-------------------
[imf_ceemdan its_ceemdan] = ceemdan(f,0.2,70,5000);

figure('Name', 'CEEMDAN�ֽ�');
subplot(6,2,1);plot(t,f,'k','linewidth',1.2);grid on;
subplot(6,2,2);hua_fft(f,fs,1);grid on;
for i = 2:6
    subplot(6,2,i*2-1);
    plot(t,imf_ceemdan(i-1,:),'k','linewidth',1.2);ylabel(['IMF',int2str(i-1)]);grid on;
    subplot(6,2,i*2);
    hua_fft(imf_ceemdan(i-1,:),fs,1);ylabel(['IMF',int2str(i-1)]);grid on;
end

figure;
for i = 1:size(imf_ceemdan,1)-5
    subplot(7,2,i*2-1);
    plot(t,imf_ceemdan(i+5,:),'k','linewidth',1.2);ylabel(['IMF',int2str(i+5)]);grid on;
    subplot(7,2,i*2);
    hua_fft(imf_ceemdan(i+5,:),fs,1);ylabel(['IMF',int2str(i+5)]);grid on;
end

corrall_ceemdan=corrcoef([f' imf_ceemdan']);
corr_ceemdan=corrall_ceemdan(1,2:end);  

%% --------------------�ع����
chonggouEMD=zeros(1,length(imf));
for i=1:size(imf,1)
    chonggouEMD=chonggouEMD+imf(i,:);
end

chonggouEEMD=zeros(1,length(imf_eemd));
for i=1:size(imf_eemd,1)
    chonggouEEMD=chonggouEEMD+imf_eemd(i,:);
end

chonggouCEEMDAN=zeros(1,length(imf_ceemdan));
for i=1:size(imf_ceemdan,1)
    chonggouCEEMDAN=chonggouCEEMDAN+imf_ceemdan(i,:);
end

error=f-chonggouEMD;
error_EEMD=f-chonggouEEMD;
error_CEEMDAN=f-chonggouCEEMDAN;
 
error1=(error)./f;
error1_EEMD=(error_EEMD)./f;
error1_CEEMDAN=(error_CEEMDAN)./f;

figure('Name', 'EMD�ع����');
plot(t,error,'k','linewidth',1.2);xlabel('Time/s');ylabel('Amplitude/V');grid on;
set(gca,'FontSize',12); 
figure('Name', 'EEMD�ع����');
plot(t,error_EEMD,'k','linewidth',1.2);xlabel('Time/s');ylabel('Amplitude/V');grid on;
set(gca,'FontSize',12); 
figure('Name', 'CEEMDAN�ع����');
plot(t,error_CEEMDAN,'k','linewidth',1.2);xlabel('Time/s');ylabel('Amplitude/V');grid on;
set(gca,'FontSize',12); 


figure('Name', 'EMD�ع��ٷֱ����');
plot(t,error1,'k','linewidth',1.2);xlabel('Time/s');ylabel('Error for EMD/%');grid on;
set(gca,'FontSize',12); 

figure('Name', 'EEMD�ع��ٷֱ����');
plot(t,error1_EEMD,'k','linewidth',1.2);xlabel('Time/s');ylabel('Error for EEMD/%');grid on;
set(gca,'FontSize',12); 

figure('Name', 'CEEMDAN�ع��ٷֱ����');
plot(t,error1_CEEMDAN,'k','linewidth',1.2);xlabel('Time/s');ylabel('Error for CEEMDAN/%');grid on;
set(gca,'FontSize',12); 


%% ______��������

figure('Name', 'EEMD��������');
boxplot(its_eemd);
figure('Name', 'CEEMDAN��������');
boxplot(its_ceemdan);


diedai_eemd=sum(its_eemd);
diedai_ceemdan=sum(its_ceemdan);

diedai_eemd=sum(diedai_eemd);
diedai_ceemdan=sum(diedai_ceemdan);

%% ----------------------����ָ��(�����ϵ��)-------------
newEMD=zeros(1,length(imf));
for i=2:size(imf,1)
    newEMD=newEMD+imf(i,:);
end

newEEMD=zeros(1,length(imf_eemd));
for i=2:size(imf_eemd,1)
    newEEMD=newEEMD+imf_eemd(i,:);
end

newCEEMDAN=zeros(1,length(imf_ceemdan));
for i=3:size(imf_ceemdan,1)
    newCEEMDAN=newCEEMDAN+imf_ceemdan(i,:);
end

x=imf_ceemdan(1,:)+imf_ceemdan(2,:);

%% --------------------------����������ֵȥ��---------------------------
 wname = 'sym4'; lev = 2;
 tree=wpdec(x,lev,wname);    


det1=wpcoef(tree,[2,3]);    
a=abs(det1);
sigma=median(a)/0.6745;     
thr = sqrt(2*log(length(x)))*sigma;        

keepapp=1;
[xd,treed,perf,perfl2]=wpdencmp(tree,'s','nobest',thr,keepapp);                                                         



figure('Name', '��������С������ֵȥ��');
subplot(221);plot(t,x,'k');xlabel('ʱ��/ms');ylabel('��ֵ/mV');grid on;
subplot(222);hua_fft(x,fs,1);xlabel('Ƶ��/kHz');ylabel('��ֵ/mV');grid on;
subplot(223);plot(t,xd,'k'); xlabel('ʱ��/ms');ylabel('��ֵ/mV');grid on;
subplot(224);hua_fft(xd,fs,1);xlabel('Ƶ��/kHz');ylabel('��ֵ/mV');grid on;
newCEEMDAN_XBB=xd+newCEEMDAN;


%% ------------- �����
 
figure('Name', 'ԭʼ�ź�+ԭʼ�źż���');
subplot(2,2,1);
plot(t,f1,'k','linewidth',1.2);xlabel('Time/ms');ylabel('Amplitude/mV');grid on;
set(gca,'FontSize',12);
subplot(2,2,2);
hua_fft(f1,fs,1);xlabel('Frequency/kHz');ylabel('Amplitude/mV');grid on; 
set(gca,'FontSize',12);
subplot(2,2,3);
plot(t,f,'k','linewidth',1.2);xlabel('Time/ms');ylabel('Amplitude/mV');grid on;
set(gca,'FontSize',12);
subplot(2,2,4);
hua_fft(f,fs,1);xlabel('Frequency/kHz');ylabel('Amplitude/mV');grid on;
set(gca,'FontSize',12);

figure('Name', '�ع�EMD');
subplot(2,2,1);
plot(t,newEMD,'k','linewidth',1.2);xlabel('Time/ms');ylabel('Amplitude/mV');grid on;
set(gca,'FontSize',12);
subplot(2,2,2);
hua_fft(newEMD,fs,1);xlabel('Frequency/kHz');ylabel('Amplitude/mV');grid on;
set(gca,'FontSize',12);

figure('Name', '�ع�EEMD');
subplot(2,2,1);
plot(t,newEEMD,'k','linewidth',1.2);xlabel('Time/ms');ylabel('Amplitude/mV');grid on;
set(gca,'FontSize',12);
subplot(2,2,2);
hua_fft(newEEMD,fs,1);xlabel('Frequency/kHz');ylabel('Amplitude/mV');grid on;
set(gca,'FontSize',12);

figure('Name', '�ع�CEEMDAN');
subplot(2,2,1);
plot(t,newCEEMDAN,'k','linewidth',1.2);xlabel('Time/ms');ylabel('Amplitude/mV');grid on;
set(gca,'FontSize',12);
subplot(2,2,2);
hua_fft(newCEEMDAN,fs,1);xlabel('Frequency/kHz');ylabel('Amplitude/mV');grid on;
set(gca,'FontSize',12);

figure('Name', '�ع�CEEMDAN+XBB');
subplot(2,2,1);
plot(t,newCEEMDAN_XBB,'k','linewidth',1.2);xlabel('Time/ms');ylabel('Amplitude/mV');grid on;
set(gca,'FontSize',12);
subplot(2,2,2);
hua_fft(newCEEMDAN_XBB,fs,1);xlabel('Frequency/kHz');ylabel('Amplitude/mV');grid on;
set(gca,'FontSize',12);

 

 
 
%% %%%%%%%%%%%%%%%%%%%-----ʵ���ź�-------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
 fs=5000; 
T=0.1;
N=fs*T;
t=1/fs:1/fs:T;
freqs = ((1/T:1/T:fs)-fs/2-1/T);
f=load('500.txt');
f=f';

%% ----------------------emd------------------
[imf,ort,nbits]=emd(f);

figure('Name', 'EMD�ֽ�');
subplot(6,2,1);plot(t,f,'k','linewidth',1);ylabel('Original signal');grid on;
subplot(6,2,2);hua_fft(f,fs,1);ylabel('Original signal');grid on;

for i = 2:6
    subplot(6,2,i*2-1);
    plot(t,imf(i-1,:),'k','linewidth',1);ylabel(['IMF',int2str(i-1)]);grid on;
    subplot(6,2,i*2);
    hua_fft(imf(i-1,:),fs,1);ylabel(['IMF',int2str(i-1)]);grid on;
end

figure;
for i = 1:size(imf,1)-5
    subplot(6,2,i*2-1);
    plot(t,imf(i+5,:),'k','linewidth',1);ylabel(['IMF',int2str(i+5)]);grid on;
    subplot(6,2,i*2);
    hua_fft(imf(i+5,:),fs,1);ylabel(['IMF',int2str(i+5)]);grid on;
end


corrall=corrcoef([f' imf']);
corr=corrall(1,2:end);       


%% ---------------------EEMD-----------------------------------
[imf_eemd its_eemd] = eemd(f,0.2,100,5000);

figure('Name', 'EEMD�ֽ�');
subplot(6,2,1);plot(t,f,'k','linewidth',1);ylabel('Original signal');grid on;
subplot(6,2,2);hua_fft(f,fs,1);ylabel('Original signal');grid on;
for i = 2:6
    subplot(6,2,i*2-1);
    plot(t,imf_eemd(i-1,:),'k','linewidth',1);ylabel(['IMF',int2str(i-1)]);grid on;
    subplot(6,2,i*2);
    hua_fft(imf_eemd(i-1,:),fs,1);ylabel(['IMF',int2str(i-1)]);grid on;
end

figure;
for i = 1:size(imf_eemd,1)-5
    subplot(6,2,i*2-1);
    plot(t,imf_eemd(i+5,:),'k','linewidth',1);ylabel(['IMF',int2str(i+5)]);grid on;
    subplot(6,2,i*2);
    hua_fft(imf_eemd(i+5,:),fs,1);ylabel(['IMF',int2str(i+5)]);grid on;
end

corrall_eemd=corrcoef([f' imf_eemd']);
corr_eemd=corrall_eemd(1,2:end);  

%% -------------------CEEMDAN-------------------
[imf_ceemdan its_ceemdan] = ceemdan(f,0.2,70,5000);

figure('Name', 'CEEMDAN�ֽ�');
subplot(6,2,1);plot(t,f,'k','linewidth',1);ylabel('Original signal');grid on;
subplot(6,2,2);hua_fft(f,fs,1);ylabel('Original signal');grid on;

for i = 2:6
    subplot(6,2,i*2-1);
    plot(t,imf_ceemdan(i-1,:),'k','linewidth',1);ylabel(['IMF',int2str(i-1)]);grid on;
    subplot(6,2,i*2);
    hua_fft(imf_ceemdan(i-1,:),fs,1);ylabel(['IMF',int2str(i-1)]);grid on;
end

figure;
for i = 1:size(imf_ceemdan,1)-5
    subplot(6,2,i*2-1);
    plot(t,imf_ceemdan(i+5,:),'k','linewidth',1);ylabel(['IMF',int2str(i+5)]);grid on;
    subplot(6,2,i*2);
    hua_fft(imf_ceemdan(i+5,:),fs,1);ylabel(['IMF',int2str(i+5)]);grid on;
end

corrall_ceemdan=corrcoef([f' imf_ceemdan']);
corr_ceemdan=corrall_ceemdan(1,2:end);  


%% -----------------�ع����
chonggouEMD=zeros(1,length(imf));
for i=1:size(imf,1)
    chonggouEMD=chonggouEMD+imf(i,:);
end

chonggouEEMD=zeros(1,length(imf_eemd));
for i=1:size(imf_eemd,1)
    chonggouEEMD=chonggouEEMD+imf_eemd(i,:);
end

chonggouCEEMDAN=zeros(1,length(imf_ceemdan));
for i=1:size(imf_ceemdan,1)
    chonggouCEEMDAN=chonggouCEEMDAN+imf_ceemdan(i,:);
end

 error=f-chonggouEMD;
 error_EEMD=f-chonggouEEMD;
 error_CEEMDAN=f-chonggouCEEMDAN;
 
 error1=(error)./f;
 error1_EEMD=(error_EEMD)./f;
 error1_CEEMDAN=(error_CEEMDAN)./f;
 

 
figure('Name', 'EMD�ع����');
plot(t,error,'k','linewidth',1.2);xlabel('Time/ms');ylabel('Amplitude/mV');grid on;
set(gca,'FontSize',12); 
figure('Name', 'EEMD�ع����');
plot(t,error_EEMD,'k','linewidth',1.2);xlabel('Time/ms');ylabel('Amplitude/mV');grid on;
set(gca,'FontSize',12); 
figure('Name', 'CEEMDAN�ع����');
plot(t,error_CEEMDAN,'k','linewidth',1.2);xlabel('Time/ms');ylabel('Amplitude/mV');grid on;
set(gca,'FontSize',12);


figure('Name', 'EMD�ع��ٷֱ����');
plot(t,error1,'k','linewidth',1.2);xlabel('Time/ms');ylabel('Error for EMD/%');grid on;
set(gca,'FontSize',12); 

figure('Name', 'EEMD�ع��ٷֱ����');
plot(t,error1_EEMD,'k','linewidth',1.2);xlabel('Time/ms');ylabel('Error for EEMD/%');grid on;
set(gca,'FontSize',12); 

figure('Name', 'CEEMDAN�ع��ٷֱ����');
plot(t,error1_CEEMDAN,'k','linewidth',1.2);xlabel('Time/ms');ylabel('Error for CEEMDAN/%');grid on;
set(gca,'FontSize',12); 



%% ----------------------��������
figure('Name', 'EEMD��������');
boxplot(its_eemd); xlabel('IMF');ylabel('Sifting iterations');set(gca,'FontSize',12);
figure('Name', 'CEEMDAN��������');
boxplot(its_ceemdan);xlabel('IMF');ylabel('Sifting iterations');set(gca,'FontSize',12);


diedai_eemd=sum(its_eemd);
diedai_ceemdan=sum(its_ceemdan);

diedai_emd=sum(nbits);
diedai_eemd=sum(diedai_eemd);
diedai_ceemdan=sum(diedai_ceemdan);

%% -------------------------����ָ�꣨�����ϵ����-------------------------------------------
newEMD=zeros(1,length(imf));
for i=2:size(imf,1)
    newEMD=newEMD+imf(i,:);
end

newEEMD=zeros(1,length(imf_eemd));
for i=2:size(imf_eemd,1)
    newEEMD=newEEMD+imf_eemd(i,:);
end

newCEEMDAN=zeros(1,length(imf_ceemdan));
for i=2:size(imf_ceemdan,1)
    newCEEMDAN=newCEEMDAN+imf_ceemdan(i,:);
end

 x=imf_ceemdan(1,:); 
%% ---------------------��������С������ֵ����--------------------------------------------------------------
 wname = 'sym4'; lev = 2;

tree=wpdec(x,lev,wname);     
det1=wpcoef(tree,[2,3]);
a=abs(det1);
sigma=median(a)/0.6745;    
thr = sqrt(2*log(length(x)))*sigma;        
keepapp=1;

xd=wpdencmp(tree,'s','nobest',thr,keepapp);



figure;
subplot(211), plot(t(1,:),(x),'k','linewidth',1), title('Original signal');grid on;
subplot(212), plot(t(1,:),(xd),'k','linewidth',1), title('Compressed signal');grid on;

newCEEMDAN_XBB=xd+newCEEMDAN;


%% ----------------����Ⱥ;��������--------------------

 
snr_1=20*log10(norm(newEMD)/norm(f-newEMD));
snr_2eemd=20*log10(norm(newEEMD)/norm(f-newEEMD));
snr_3ceemdan=20*log10(norm(newCEEMDAN)/norm(f-newCEEMDAN));
snr_4ceemdan_XBB=20*log10(norm(newCEEMDAN_XBB)/norm(f-newCEEMDAN_XBB));


rmse_1=sqrt(norm(f-newEMD)/length(f));
rmse_2eemd=sqrt(norm(f-newEEMD)/length(f));
rmse_3ceemdan=sqrt(norm(f-newCEEMDAN)/length(f));
rmse_4ceemdan_XBB=sqrt(norm(f-newCEEMDAN_XBB)/length(f));



figure('Name', 'ԭʼ�ź�');
subplot(2,2,1);
plot(t,f,'k','linewidth',1.1);xlabel('Time/ms');ylabel('Amplitude/mV');grid on;set(gca,'FontSize',12); 
% axis([0.02 0.06 -0.01 0.01]);
subplot(2,2,2);
hua_fft(f,fs,1);xlabel('Frequency/kHz');ylabel('Amplitude/mV');grid on;set(gca,'FontSize',12); 

figure('Name', 'EMD�ֽ�');
subplot(2,2,1);
plot(t,newEMD,'k','linewidth',1.1);xlabel('Time/ms');ylabel('Amplitude/mV');grid on;set(gca,'FontSize',12); 
% axis([0.02 0.06 -0.01 0.01]);
subplot(2,2,2);
hua_fft(newEMD,fs,1);xlabel('Frequency/kHz');ylabel('Amplitude/mV');grid on;set(gca,'FontSize',12); 

figure('Name', 'EEMD�ֽ�');
subplot(2,2,1);
plot(t,newEEMD,'k','linewidth',1.1);xlabel('Time/ms');ylabel('Amplitude/mV');grid on;set(gca,'FontSize',12); 
% axis([0.02 0.06 -0.01 0.01]);
subplot(2,2,2);
hua_fft(newEEMD,fs,1);xlabel('Frequency/kHz');ylabel('Amplitude/mV');grid on;set(gca,'FontSize',12); 

figure('Name', 'CEEMDAN�ֽ�');
subplot(2,2,1);
plot(t,newCEEMDAN,'k','linewidth',1.1);xlabel('Time/ms');ylabel('Amplitude/mV');grid on;set(gca,'FontSize',12); 
% axis([0.02 0.06 -0.01 0.01]);
subplot(2,2,2);
hua_fft(newCEEMDAN,fs,1);xlabel('Frequency/kHz');ylabel('Amplitude/mV');grid on;set(gca,'FontSize',12); 

figure('Name', 'CEEMDAN�ֽ�+XBB');
subplot(2,2,1);
plot(t,newCEEMDAN_XBB,'k','linewidth',1.1);xlabel('Time/ms');ylabel('Amplitude/mV');grid on;set(gca,'FontSize',12); 
% axis([0.02 0.06 -0.01 0.01]);
subplot(2,2,2);
hua_fft(newCEEMDAN_XBB,fs,1);xlabel('Frequency/kHz');ylabel('Amplitude/mV');grid on;set(gca,'FontSize',12); 


 
 