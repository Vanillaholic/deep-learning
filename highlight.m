classdef highlight
    % HighlightTF: 模拟潜艇的亮点转移函数
    % Properties:
    %   c: 声速
    %   fs: 采样频率
    %   N: 频域的取值点数
    %   theta: 入射角度
    %   L: 潜艇的各部位距离尾部的长度
    %   b: 潜艇各部位的反射系数
    %   r: 亮点的半径
    % Methods:
    %   compute_transfer_function: 计算亮点转移函数
    %   getImpulseResponse: 获取脉冲响应
    
    properties
        c        % 声速
        fs       % 采样频率
        N        % 频域的取值点数
        theta    % 入射角度
        L        % 潜艇的各部位距离尾部的长度
        b        % 潜艇各部位的反射系数
        r        % 亮点的半径
    end
    
    methods
        % 构造函数
        function obj = highlight(c, fs, N, theta, L, b, r)
            if nargin > 0
                obj.c = c;
                obj.fs = fs;
                obj.N = N;
                obj.theta = theta;
                obj.L = L;
                obj.b = b;
                obj.r = r;
            end
        end
        
        % 计算亮点转移函数
        function [TF2, IR_Highlight] = get_TF_and_IR(obj)
            tauHighLight = zeros(6, 1);
            f = -obj.fs/2:obj.fs/obj.N:obj.fs/2 - obj.fs/obj.N;

            % 计算亮点的时间延迟
            for j = 1:6
                term1 = (obj.L(j) * cosd(obj.theta))^2;
                term2 = (obj.L(j) * sind(obj.theta))^2;
                numerator = 2 * sqrt(term1 + term2);
                tauHighLight(j) = numerator / obj.c;
            end

            % 计算亮点之间的几何关系
            L_ij = zeros(6, 6);
            h_ij = zeros(6, 6);
            S_ij = zeros(6, 6);
            for i = 1:6
                for j = 1:6
                    L_ij(i, j) = abs(obj.L(j) - obj.L(i));
                    h_ij(i, j) = (L_ij(i, j) - obj.r(i) / (sind(obj.theta) + eps)) * sind(obj.theta);
                    gamma = acos(h_ij(i, j) / obj.r(i));
                    S_ij(i, j) = obj.r(i)^2 / 2 * (2 * gamma) - h_ij(i, j) * sqrt(obj.r(i)^2 - h_ij(i, j)^2);
                end
            end

            % 计算隐蔽系数
            C_ij = zeros(6, 6);
            for i = 1:6
                for j = 1:6
                    if abs(L_ij(i, j) * sind(obj.theta)) < obj.r(i) + obj.r(j)
                        C_ij(i, j) = S_ij(i, j) / (pi * obj.r(i)^2);
                    else
                        C_ij(i, j) = 0;
                    end
                end
            end
            C_ij(1:size(C_ij, 1) + 1:end) = 0;

            % 计算每个亮点的反射系数和目标强度
            B = zeros(1, 6);
            C = zeros(1, 6);
            TS = zeros(1, 6);
            A = zeros(1, 6);
            for i = 1:6
                C(i) = max(C_ij(i, :));
                B(i) = obj.b(i) * (1 - C(i));
                TS(i) = 10 * log10(obj.r(i)^2 / 4) + 10 * log10(B(i));
                A(i) = 10^(TS(i) / 20);
            end

            % 计算亮点模型的转移函数
            highlightTF = zeros(6, obj.N);
            phi = pi;
            for i = 1:6
                highlightTF(i, :) = A(i) * exp(-1j * (2 * pi * f) * tauHighLight(i)) .* exp(1j * phi);
            end
            TF2 = sum(highlightTF, 1);

            % 计算脉冲响应
            IR_Highlight = ifft(TF2);
        end

        % 获取脉冲响应
        function IR_Highlight = getImpulseResponse(obj)
            [~, IR_Highlight] = obj.get_TF_and_IR();
        end
    end
end
