function TF2 = get_highlight_TF(c,fs,N,theta,L,b,r)
% 此函数仿真潜艇的亮点转移函数
%共有6个两点
% c：声速
% fs：采样频率
%  N: 频域的取值点数
% theta:入射角度
% L：潜艇的各部位距离尾部的长度
%b： 潜艇各部位的反射系数
% r：亮点的半径

tauHighLight = zeros(6,1);

f = -fs/2:fs/N:fs/2-fs/N;

for j = 1:6
     term1 = (L(j)*cosd(theta)).^2;
     term2 = (L(j)*sind(theta))^2;
     numerator = 2*sqrt(term1 + term2);
     tauHighLight(j) = numerator/c;  %此时c为1530m/s 
end


%

L_ij = zeros(6,6);
h_ij = zeros(6,6);
S_ij = zeros(6,6);
%gamma为图中的alpha
for i =1:6
    for j = 1:6
        L_ij(i,j) =abs( L(j) - L(i) );
        h_ij(i,j) = (L_ij(i,j) - r(i)/(sind(theta)+eps) ).*sind(theta);
        gamma = acos(h_ij(i,j)/r(i));
        S_ij(i,j) = r(i).^2 /2.*(2*gamma) - h_ij(i,j)*sqrt(r(i).^2 - h_ij(i,j).^2);
    end
end

% 计算隐蔽系数
C_ij = zeros(6,6);
for i = 1:6
    for j = 1:6
        if abs( L_ij(i,j)*sind(theta) ) < r(i) + r(j)   %传播方向上的投影长度小于亮点半径之和
            C_ij(i, j) = S_ij(i,j)/(pi  *  (r(i).^2)  ); 
        else
            C_ij(i, j) = 0;
        end
    end
end
C_ij(1:size(C_ij,1)+1:end) = 0;  % 将 C_ij 的对角元素设为 0,因为隐蔽系数的计算不考虑亮点自己本身
% 计算每个亮点的实际反射系数
B = zeros(1, 6);
for i = 1:6
    C(i) = max(C_ij(i, :));
    B(i) = b(i) .* (1 - C(i));
    % 计算目标强度
    TS(i) = 10 * log10(r(i).^2 / 4) + 10 * log10(B(i));
    
    % 计算亮点子回波幅度
    A(i) = 10.^(TS(i) / 20);
end


%计算亮点模型的转移函数
highlightTF = zeros(6,N);
temp = zeros(1,N);
phi = pi;
for i = 1:6%计算每个部位的转移函数
    highlightTF(i,:) = A(i)*exp(-1j*(2*pi*f)*tauHighLight(i)).*exp(1j*phi);
end
TF2 = sum(highlightTF,1);%将转移函数相加


end