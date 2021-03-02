clear
close all

%% Digital image transformation
% 1. If the input digital image is a colour image, say I, convert it into an image in grey level, say G ; and Binarisation
I = imread('5.jpg');
G = rgb2gray(I);

figure()
imshow(I)
figure()
imshow(G)
X = im2bw(G,0.7);
%X = imbinarize(rgb2gray(imread('9.jpg')));
% 2. If necessary, compute the transpose of G ;
% 3. If possible, define a sub-zone of the image in which text regions are going to be looked for.

figure()
imshow(X)
%% Enhancement of text region patterns
% multi-resolution process (= Rezize the image)
M = 0.125;
Xresized = imresize(X, M, 'nearest');
%Xdouble = im2double(Xresized);
%Xbin = Xdouble > 0.7;

figure()
imshow(Xresized)
Xbin=Xresized;

%% Potential text region localization: Binary image with potential text regions delimitated by white blocks
% M5 implementation
[l,c] = size(Xbin);
ind = zeros(2,l);


% M1
for i = 1:l
    for j = 1:c
        if Xbin(i,j) == 1 && ind(1,i) == 0
            ind(1,i) = j;
        end
        if Xbin(i,j) == 1
            ind(2,i) = j;
        end
    end
    if ind(1,i)*ind(2,i) > 0 && ind(2,i)-ind(1,i) < 3*c/4
        Xbin(i,ind(1,i):ind(2,i)) = 1;
    end
    if ind(1,i)*ind(2,i) > 0 && ind(2,i)-ind(1,i) >= 3*c/4
        Xbin(i,ind(1,i):ind(2,i)) = 0;
    end
    if ind(1,i) == ind(2,i) && ind(1,i) ~= 0
        Xbin(i,ind(1,i)) = 0
    end
end

% implémentation M2 & M3

Xbincopy = zeros(l,c);

while sum(sum((Xbincopy ~= Xbin)))
    Xbincopy = Xbin
    for i = 2:l
        for j = 1:c-1
            if Xbin(i-1,j) && Xbin(i,j+1)
                Xbin(i-1:i,j:j+1) = 1;
            end
            if Xbin(i-1,j+1) && Xbin(i,j)
                Xbin(i-1:i,j:j+1) = 1;
            end
        end
    end
end



figure()
imshow(Xbin)


Xbinpad = padarray(Xbin,[1 1],0,'both');
figure()
imshow(Xbinpad)
[l1,c1] = size(Xbinpad);
for i = 2:l1-1
    for j = 2:c1-1
        if Xbinpad(i,j) && not(Xbinpad(i-1,j-1) || Xbinpad(i-1,j) || Xbinpad(i-1,j+1) || Xbinpad(i,j-1) || Xbinpad(i,j+1) || Xbinpad(i+1,j-1) || Xbinpad(i+1,j) || Xbinpad(i+1,j+1))
            Xbinpad(i,j) = 0;
        end
    end
end

Xbin = Xbinpad(2:end-1,2:end-1);


figure()
imshow(Xbin)

%% Selection of effective text regions
% Background pixels separation

histI = imhist(G);
figure()
plot(histI);

threshold_bkpxsep = 0.02;
L = 256;
u = L;
sum(histI);
sum(histI(u:L));
while sum(histI(u:L-1)) < threshold_bkpxsep * sum(histI)
    sum(histI(u:L));
    u = u - 1;
end

for i=u:L-1
    histI(L) = histI(L) + histI(i);
    histI(i)=0;
end
figure()
plot(histI);

% Effective text region filtering

% Ir = imresize(I, M, 'nearest');
%  Ir(not(Xbin))=0
Xbinr = imresize(Xbin, 1/M, 'nearest');
nb_zone = 10;
figure()
imshow(Xbinr)
ind=0;
G = padarray(G,size(Xbinr)-size(G),1,'post');
%I(not(Xbinr));
text_regions_rect = [];
[l,c] = size(Xbinr);
y=1;
while y < l
    if Xbinr(y,:) == zeros(1,c)
        y = y + 1;
    else
        ind = ind + 1;
        h=1;
        term=false;
%         while not(term)
%             if Xbinr(y+h,:) == zeros(1,c)
%                 term = true;
%             else
%                 h = h + 1;
%             end
%         end
        while (y+h<l) && sum(Xbinr(y+h,:) == Xbinr(y,:)) == c
            h = h + 1;
        end
        x=1;
        w=1;
        while not(Xbinr(y, x))
            x = x + 1;
        end
        while Xbinr(y, x+w)
            w = w +1;
        end

        fprintf('zone [%i %i %i %i] \n', x, y, w, h);
        I2 = imcrop(G,[x y w h]);
        %G2=I;
        subplot(nb_zone,2,2*ind-1), imshow(I2)

        histI2 = imhist(I2);
        for i=u:L-1
            histI2(L) = histI2(L) + histI2(i);
            histI2(i) = 0;
        end
        %subplot(nb_zone,2,2*ind), plot(histI2);
        
        
        histI2copie = histI2;
        for i = 6:u
            histI2(i) = mean(histI2copie(i-5:i));
        end
        subplot(nb_zone,2,2*ind), findpeaks(histI2,'NPeaks',5,'SortStr','descend','MinPeakProminence',0)
        [PKS, LOCS] = findpeaks(histI2,'NPeaks',5,'SortStr','descend','MinPeakProminence',0);
        
        [tempmax,tempind] = max(PKS) 
        
        if histI2(256) > PKS(1) && 255-LOCS(tempind) > 0.25*255
                fprintf("texte detecté !\n");
                title('texte detecté !');
                text_regions_rect = [ text_regions_rect; [x y w h] ];
        else
                fprintf("pas de texte détecté !\n");
                title('pas de texte détecté !');
        end
        

%         %[M1,P1] = max(histI2);
%         P1 = find(histI2 == max(histI2(:)));
%         histI2(P1) = 0;
%         P2 = find(histI2 == max(histI2(:)));
%         dist = min([abs(max(P1)-min(P2)), abs(min(P1)-max(P2))]);
%         
%         threshold_spectre = .15;
%         if dist > threshold_spectre*length(histI)
%             fprintf("texte detecté !\n");
%             title('texte detecté !');
%             text_regions_rect = [ text_regions_rect; [x y w h] ];
%         else
%             fprintf("pas de texte détecté !\n");
%             title('pas de texte détecté !');
%         end
        
        y = y + h;
    end
end


%% Improving text region localization
figure, imshow(I);
s = size(text_regions_rect);
for rect_i=1:s(1)
    % Horizontal delimitation of text region boundaries
    
    text_regions_rect(rect_i,:)
    Rh_lg_i = text_regions_rect(rect_i,2); % representative horizontal line
    max_i = sum(G(Rh_lg_i,:)>u);
    for i = text_regions_rect(rect_i,2):text_regions_rect(rect_i,2)+text_regions_rect(rect_i,4)
        if sum(G(i,:)>u) > max_i
            Rh_lg_i=i;
            max_i = sum(G(i,:)>u);
        end
    end
    max_i
    
    [l,c] = size(G);
    dm = 0;
    while sum((G(Rh_lg_i-dm,:)>u) & (G(Rh_lg_i-dm-1,:)>u))
         dm = dm + 1;
    end
    Rh_lg_i
    dp = 0;
    while i+dp+1<=l && sum((G(Rh_lg_i+dp,:)>u) & (G(Rh_lg_i+dp+1,:)>u))
         dp = dp + 1;
    end
    dm
    dp
    y_rect = Rh_lg_i - dm - 1;
    h_rect = 1 + dp + dm + 2;
    % Vertical delimitation of texts region boundaries
    x_rect = 1;
    while G(Rh_lg_i, x_rect) < u
        x_rect = x_rect + 1;
    end

    x_max_rect = c;
    while G(Rh_lg_i, x_max_rect) < u
        x_max_rect = x_max_rect - 1;
    end
    w_rect = x_max_rect - x_rect;

    rectangle('position',[x_rect y_rect w_rect h_rect],'edgecolor','k','LineWidth',4)
    rectangle('position',[x_rect y_rect w_rect h_rect],'edgecolor','w','LineWidth',1)
    %figure, imshow(imcrop(G, [x_rect y_rect w_rect h_rect]))
end