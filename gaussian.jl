using DrWatson
import Pkg; Pkg.activate("/storage/home/suv87/work/julia/invariance")

using Plots, Distributions
using QuadGK
using Ripserer

pgfplotsx()
gr()

plot(x -> x^2, -2, 2)




pltt(θ; kwargs...) = begin
    f(θ) = (x, y) -> ((cos(θ) * quantile(Normal(), x)) + (sin(θ) * quantile(Normal(), y)))^2
    seq = [0:0.01:0.25...; 0.26:0.1:0.74...; 0.75:0.01:1...]
    plot(
        seq, seq, f(θ),
        st=:surface,
        title="θ = $(round(θ, digits=3))",
        zlim=(0, 10),
        fa=0.5; kwargs...
    )
end

pltt(0.1)




# Normal Distribution
@gif for θ ∈ range(0, π, length=100)
    seq = [0:0.01:0.25...; 0.26:0.1:0.74...; 0.75:0.01:1...]
    f(θ) = (x, y) -> ((cos(θ) * quantile(Normal(), x)) + (sin(θ) * quantile(Normal(), y)))^2
    plot(
        seq, seq, f(θ),
        st=:surface,
        title="θ = $(round(θ, digits=3))",
        zlim=(0, 10),
        fa=0.5,
        # c=palette(:viridis, rev=false)
    )
end

# (a + b) = 1 (x/a) and (-x/b)

# Gaussian Mixture Distribution

w = 0.7
D = MixtureModel([Gamma(10, 5), Gamma(10, 2)], [w, 1 - w])
g(x) = pdf(D, x)
f(θ) = x -> x ≥ 0 ? 0.5 * g(x / θ) : 0.5 * g(-x / (2 - θ))

a = @animate for θ ∈ range(0.1, 1.9, length=50)
    t = 0.007
    xseq = -200:0.01:200
    ys = map(f(θ), xseq)
    yss = map(x -> x < t ? 0.0 : x, ys)

    h(x) = f(θ)(x) < t ? 0.0 : f(θ)(x)
    area = quadgk(h, -200, 200, rtol=1e-5)[1]

    plt = plot(xseq, yss, fill=0, la=0, fa=0.5, label="mass = $(round(area, digits=3))")
    plt = hline(plt, [t], ls=:dash, lw=1, label="")
    plt = plot(plt, xseq, ys, c=:black, lw=2, label="")


    tit = "(a, b) = $(round.((1/θ, 1/(2-θ)), digits=2))"

    plot(plt, title=tit, size=(900, 300))
end
gif(a, fps=10)


begin
    # θ = 0.33333
    θ = 1.0
    # θ = 1.5

    t = 0.006
    xseq = -200:0.01:200
    ys = map(f(θ), xseq)
    yss = map(x -> x < t ? 0.0 : x, ys)

    h(x) = f(θ)(x) < 0.005 ? 0.0 : f(θ)(x)
    area = quadgk(h, -200, 200, rtol=1e-3)[1]

    plt = plot(xseq, yss, fill=0, la=0, fa=0.5, label="mass = $(round(area, digits=3))")
    plt = hline(plt, [t], ls=:dash, lw=1, label="")
    plt = plot(plt, xseq, ys, c=:black, lw=2, label="")


    tit = "(a, b) = $(round.((1/θ, 1/(2-θ)), digits=2))"

    plot(plt, title=tit, size=(400, 400))

    # savefig(plotsdir("gaussian_mixture_$(round(θ, digits=1)).pdf"))
end




##############################################

using LinearAlgebra


plt_colorbar(z; kwargs...) = scatter(
    [0, 0], [0, 1],
    zcolor=[0, 1], clims=extrema(z), xlims=(1, 1.1),
    xshowaxis=false, yshowaxis=false, axis=false, ticks=false,
    label="", c=:viridis, colorbar_title="cbar", grid=false; kwargs...
)

h2 = scatter([0, 0], [0, 1], zcolor=[0, 1], clims=(0, 1),
    xlims=(1, 1.1), xshowaxis=false, yshowaxis=false, label="", c=:viridis, colorbar_title="cbar", grid=false, axis=false, ticks=false)


function f(ρ)
    function foo(x, y)
        r = norm([x; y])
        θ = atan(y / x)
        return (2π)^-1 * exp(-r^2 / (1 * (1 + ρ * cos(θ))))
    end
    return foo
end

g(ρ) = (x, y) -> f(ρ)(x, y) >= exp(-0.5) / 2π


gr(c=palette(:inferno))
plt = plot(
    contourf(-1:0.01:1, -1:0.01:1, f(-0.99), colorbar=false),
    contourf(-1:0.01:1, -1:0.01:1, f(-0.5), colorbar=false),
    contourf(-1:0.01:1, -1:0.01:1, f(0.99), colorbar=false),
    scatter([0, 0], [0, 0.01], zcolor=[0, 0.155], clims=(0, 0.155), ticks=false,
        xlims=(1, 1.0000001), axis=false, label="", colorbar_title="", grid=false),
    layout=(@layout [grid(1, 3) a{0.1w}]), size=(900, 250),
    levels=9, lw=0.1, link=:all, title=["ρ=–0.99" "ρ=–0.5" "ρ=+0.99" ""]
)
savefig(plotsdir("contours.pdf"))


plot(x -> 1 / (1 - 0.99 * cos(x)), 0, π / 2)
plot!(x -> 1 / (1 + 0.99 * cos(x)), 0, π / 2)

