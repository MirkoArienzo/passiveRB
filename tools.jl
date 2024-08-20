using TensorKit, TensorOperations, SUNRepresentations, Permanents, Memoize

rowsum(m, l) = l == 0 ? 0 : sum(m[k, l] for k in 1:l)

τ(n,m) = SUNIrrep(Tuple([[n];fill(0,m-1)]))

λ(k,m) = SUNIrrep(Tuple([[2*k];fill(k,m-2);[0]]))

sλ(k, m) = 1 / ( 1 + k / (k+m-1) ) / binomial(Int128(k+m-1),m-1)

inv_sλ(k, m) = ( 1 + k / (k+m-1) ) * binomial(Int128(k+m-1),m-1)

function maxGTpat(GTpat)
    d = length(weight(GTpat))
    maxGTpat = Base.setindex(GTpat, GTpat[1, d], 1, 1)
    for l in 2:d
        for k in 1:l
            maxGTpat = Base.setindex(maxGTpat, GTpat[k, d], k, l)
        end
    end
    return maxGTpat
end

φ₀(GTpat) = sum([rowsum(GTpat,i) for i in 1:(length(weight(GTpat))-1)])

φ(GTpat) = φ₀(GTpat) - φ₀(maxGTpat(GTpat))

function FockState(lst)
    m = length(lst)
    n = sum(lst)
    GTpat = Base.setindex(collect(basis(τ(n,m)))[1], lst[1], 1, 1)
    for l in 2:m
        GTpat = Base.setindex(GTpat, sum(lst[1:l]), 1, l)
    end
    for l in 2:m, k in 2:l
        GTpat = Base.setindex(GTpat, 0, k, l)
    end
    return GTpat
end

function GTindex(GTpat)
    d = length(weight(GTpat))
    λ = Tuple([GTpat[i,d] for i in 1:d])
    return findfirst(item -> item == GTpat, collect(basis(SUNIrrep(λ))))
end
        
function dualGTpat(GTpat)
    d = length(weight(GTpat))
    dualGTpat = Base.setindex(GTpat, GTpat[1, d] - GTpat[1, 1], 1, 1)
    for l in 2:d
        for k in 1:l
            dualGTpat = Base.setindex(dualGTpat, GTpat[1, d] - GTpat[k, l], l-k+1, l)
        end
    end
    return dualGTpat
end

@memoize function g(l, k, X_gtp, N_gtp)
    n = sum(weight(N_gtp))
    m = length(weight(N_gtp))
    
    X = GTindex(X_gtp)
    XB = GTindex(dualGTpat(X_gtp))
    N = GTindex(N_gtp)
    NB = GTindex(dualGTpat(N_gtp))
    
    tNk = CGC(τ(n,m), conj(τ(n,m)), λ(k,m))[N,NB,:,1]
    tXk = CGC(τ(n,m), conj(τ(n,m)), λ(k,m))[X,XB,:,1]
    tNl = CGC(τ(n,m), conj(τ(n,m)), λ(l,m))[N,NB,:,1]
    tXl = CGC(τ(n,m), conj(τ(n,m)), λ(l,m))[X,XB,:,1]
    tkkl = CGC(λ(k,m), λ(k,m), λ(l,m))
    
    return @tensor tNk[M] * tXk[L] * tNl[R] * tXl[R′] * tNk[M′] * tXk[L′] * tkkl[M,M′,R,r] * tkkl[L,L′,R′,r]
end

@memoize function moment1(k, N)
    n = sum(weight(N))
    m = length(weight(N))
    CGvec = CGC(τ(n,m), conj(τ(n,m)), λ(k,m))[GTindex(N),GTindex(dualGTpat(N)),:,1]
    return @tensor CGvec[M] * CGvec[M]
end

@memoize function moment2(k, N)
    n = sum(weight(N))
    m = length(weight(N))
    s_array = zeros(Float64, Threads.nthreads())
    Threads.@threads for X in collect(basis(τ(n,m)))
        for l in 0:min(n,2*k) 
            s_array[Threads.threadid()] += ( g(l, k, X, N) / dim(λ(l,m)) ) * (-1) ^ (φ(X))
        end
    end
    s = sum(s_array)
    return ( s * (inv_sλ(k, m) ^ 2) ) * (-1)^(φ(N))
end

function symmmetric_permanent(X,Y,g)
    Xw = weight(X)
    Yw = weight(Y)
    n = sum(weight(X))
    mat = zeros(ComplexF64,n,n)
    row_indices = []
    col_indices = []
    for (i, mult) in enumerate(Xw)
        append!(row_indices, repeat([i], mult))
    end
    for (j, mult) in enumerate(Yw)
        append!(col_indices, repeat([j], mult))
    end
    return Permanents.naive(g[row_indices,col_indices]) / sqrt(prod(factorial.(Xw)) * prod(factorial.(Yw)) )
end

function filter_function(k,N,X,g_mat)
    n = sum(weight(N))
    m = length(weight(N))
    s_array = zeros(ComplexF64, Threads.nthreads())
    Threads.@threads for Y in collect(basis(τ(n,m)))
        t = CGC(τ(n,m), conj(τ(n,m)), λ(k,m))[GTindex(N),GTindex(dualGTpat(N)),:,1]
        t12 = CGC(τ(n,m), conj(τ(n,m)), λ(k,m))[GTindex(Y),GTindex(dualGTpat(Y)),:,1]
        s_array[Threads.threadid()] += (-1)^(φ(Y)) * (@tensor t[M] * t12[M]) * abs(symmmetric_permanent(X,Y,g_mat))^2
    end
    s = sum(s_array)
    return s * inv_sλ(k, m) * (-1)^(φ(N))
end