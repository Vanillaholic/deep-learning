function seqs = lkasami(oct,gen)

% function seqs = lkasami(oct)
% This function generates Large Kasami sequences.
% oct generate an m-sequence of period 2^n-1
% oct should be in oct form. n should be even.
% The first (N+1)^(0.5) sequences are the small set Kasami sequences. 
% reference: Dilip V. Sarwate and Michael B. Pursley, 1980.

% Qinghua Zhao
% Aug. 2001 at UCSD

%1.先生成m序列
seqs = [];
u = m_seq(oct,gen);
N = size(u, 2); 
n = log2(N+1);
if mod(n, 2) ~= 0
	fprintf(1, 'No Kasami sequence exist for generator of this order.\n');
	beep;
	return;
end 

t = 1 + 2^(n/2+1); % no need to take floor since it will be an interger.
v = u(mod(0:t:N*t-1, N)+1);

% generate w
N_2 = 2^(n/2)+1;
w = u(1:N_2:end);
if sum(w)==0 % if all zero sequence, then pick another one. It won't be all one sequence since it can only be an m-sequence. 
	w = u(2:N_2:end);
end
w = ones(N_2, 1)*w;
w = reshape(w', 1, N);

fprintf(1, 'First %d sequences are small set Kasami\n', N_2-1); 

if mod(n, 4) ~= 0
	% generate gold sequences 
	G_uv = [u; v];
	for i = 1:N
	        G_uv = [G_uv; xor(u, v)];
        	v = rshift(v);
	end

	% generate kasami sequences
	% the first 2^(n/2) sequences are the small set kasami sequences.
	for k = 1:size(G_uv, 1)
		seqs = [seqs; G_uv(k,:)];
		for i = 1:N_2-2
        		seqs = [seqs; xor(G_uv(k,:), w)];
        		w = rshift(w);
		end
	end
else
	% generate gold like sequences
	v1 = u(mod(0:t:N*t-1, N)+2);
	v2 = u(mod(0:t:N*t-1, N)+3);
	Ht_u = u;
	for i = 1:N/3
 		Ht_u = [Ht_u; xor(u, v); xor(u, v1); xor(u, v2)];	
        	v = rshift(v); v1 = rshift(v1); v2 = rshift(v2);
	end

	% generate kasami sequences
	% the first 2^(n/2) sequences are the small set kasami sequences.
	for k = 1:size(Ht_u, 1)
		seqs = [seqs; Ht_u(k,:)];
		for i = 1:N_2-2
        		seqs = [seqs; xor(Ht_u(k,:), w)];
        		w = rshift(w);
		end
	end
	for i = 1:(N_2-2)/3
		seqs = [seqs; xor(v, w); xor(v1, w); xor(v2, w)];
        	w = rshift(w);
	end


end

% 该函数用于生成m序列，基于给定的生成器oct。
% 参考文献：Dilip V. Sarwate和Michael B. Pursley，1980年。第599页。图1。

function seq = m_seq(oct,gen)

% gen = oct2gen(oct);     % oct2gen may be removed from future version
%1.查找非零索引，并将序列从该位置之后的位置提取出来
s = min(find(gen));
gen = gen(s+1:end);

%2.计算生成器的位数和序列的长度
n = size(gen, 2);
N = 2^n-1;

%3.将生成器反转
gen = fliplr(gen);

seq = zeros(1, n); seq(n) = 1;
for i=1:N-n
        next_bit = mod(sum(seq(i:i+n-1)&gen), 2);
        seq = [seq, next_bit];
end

return

function y = rshift(x)
y = [x(:, end), x(:, 1:end-1)];
return

