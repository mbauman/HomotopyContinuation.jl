using Pkg
Pkg.activate(@__DIR__)

using HomotopyContinuation, Test, LinearAlgebra
const HC = HomotopyContinuation

F = let
    @var x[1:2] a[1:5] c[1:6] y[1:2, 1:5]

    #tangential conics
    f =
        a[1] * x[1]^2 +
        a[2] * x[1] * x[2] +
        a[3] * x[2]^2 +
        a[4] * x[1] +
        a[5] * x[2] +
        1
    ∇ = differentiate(f, x)
    #5 conics
    g =
        c[1] * x[1]^2 +
        c[2] * x[1] * x[2] +
        c[3] * x[2]^2 +
        c[4] * x[1] +
        c[5] * x[2] +
        c[6]
    ∇_2 = differentiate(g, x)
    #the general system
    #f_a_0 is tangent to g_b₀ at x₀
    function Incidence(f, a₀, g, b₀, x₀)
        fᵢ = f(x => x₀, a => a₀)
        ∇ᵢ = ∇(x => x₀, a => a₀)
        Cᵢ = g(x => x₀, c => b₀)
        ∇_Cᵢ = ∇_2(x => x₀, c => b₀)
        [fᵢ; Cᵢ; det([∇ᵢ ∇_Cᵢ])]
    end
    @var v[1:6, 1:5]
    F = vcat(map(i -> Incidence(f, a, g, v[:, i], y[:, i]), 1:5)...)
    System(F, [a; vec(y)], vec(v))
end

p = Complex{Float64}[0.7409657355057058 - 0.3326704870405587im, -0.30343376263156574 - 0.6796065707578953im, -1.3798593495885692 - 0.36670321912221254im, 0.8750974468522109 - 0.42208422943347895im, 0.0017452349083929572 - 0.06195595247308483im, 0.5543836956815789 - 0.6542430892213258im, 0.27020874003334305 - 0.593215257720508im, 0.3463050394133096 + 0.7138156023032164im, -0.08966901318391826 + 1.013410704730413im, -0.794450134609723 + 0.08606287009829472im, 0.7383115760683732 - 0.6303538756408905im, -0.7732999902981492 + 1.1205969517136622im, 0.4715586590117718 - 0.259553757004003im, -0.6059236674190519 + 0.2197767977020651im, -0.6237858637342937 + 0.1733090402556765im, -0.593495649170219 + 0.21419510621102178im, -0.8390819388474703 + 1.230903390977708im, 0.47339435770330657 + 0.530328092354497im, 0.5456360687179636 + 1.4783267407512763im, 0.3768087111715607 + 0.31971983532042053im, 0.5961673329747379 + 0.21929488563443858im, -0.45520833284019996 - 0.33116341727977044im, -0.43030049985295504 - 0.7976060027356192im, -0.39608251944221107 + 0.4660007624390357im, 0.47207651238321724 + 0.03340697340493345im, -1.2178311514592401 - 0.4376518485417597im, -0.6570319310908632 - 1.8278498464288913im, -1.2034588631579957 - 0.31661802266646366im, 1.2162329306080075 - 0.04746989063175015im, -0.7243212329301247 - 0.33433168511913536im]

res = solver_startsolutions(F, target_parameters = p, seed = 0x85d6db88,
        endgame_options = (max_endgame_steps = 3000, val_at_infinity_tol = 1e-4, min_coord_growth = 400))



# G = System(F(variables(F), p), variables(F))
supp, _ = support_coefficients(F)
G = HC.polyhedral_system(supp)
c = randn(ComplexF64, 140)

gen_res = solve(G, target_parameters = c, seed = 0x85d6db88,)

solver, starts = solver_startsolutions(G, target_parameters = c, seed = 0x85d6db88,)
S = collect(starts)

track(solver, S[1051]; debug = true)

Y = HC.solve(starts.BSS, starts.support, starts.start_coefficients, S[1051][1])
y = Y[:,1]
[prod(y .^ c) - b for (c, b) in zip(eachcol(starts.BSS.A), starts.BSS.b)]


U = copy(starts.BSS.H)

starts.BSS

starts.BSS.H_big == starts.BSS.H






setprecision(BigFloat, 1024)












S[1051]




HC.evaluate!(zeros(ComplexF64, 15), solver.trackers[1].toric_tracker.homotopy, S[1051][2], 0.0)

solver.trackers[1].toric_tracker.homotopy.system




res = solve(F, target_parameters = p, seed = 0x85d6db88,
        endgame_options = (max_endgame_steps = 3000, val_at_infinity_tol = 1e-4, min_coord_growth = 400))

res[1]

MF = ModelKitSystem(F)



results = filter(r -> r.return_code == :polyhedral_failed, path_results(res))

sort(map(results) do r
    cond(HC.jacobian(ModelKitSystem(F), solution(r), p))
end)


multiplicities(solutions(results))

sort(results; by= cond)




tracker = HC.parameter_homotopy(F; generic_parameters = p,
 endgame_options = (endgame_start = 0.0,))

track(tracker, x)




tree = HC.VoronoiTree(first(data), 1)
HC.add!(tree, data[1], 1, 1e-12)

Juno.@enter HC.add!(tree, data[2], 2, 1e-12)



tree





for (i, d) in enumerate(data)


    insert!(tree, d, i)
end
@test all(i -> HC.search_in_radius(tree, data[i], 1e-12) == i, 1:1000)
