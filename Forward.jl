#!/usr/bin/env julia
# -----------------------------------------------------------------------------
# ForwardSynthesis.jl
#
# A scriptified version of Forward.ipynb:
#   - Reads Bifrost/Multi3D atmosphere
#   - Builds LTE populations via Saha-Boltzmann
#   - Remaps atmosphere + populations to a new column-mass (cmass) scale
#   - Two synthesis modes:
#       (A) "ml"     : NLTE populations = (predicted departure coeffs) * (LTE pops)
#       (B) "bifrost": NLTE populations read from Multi3D out_pop (and remapped)
#   - Synthesizes 1D line profiles for all columns (nx, ny) using Muspel
#   - Writes intensity + wavelength to an HDF5 file
#   - Writes two diagnostic plots (PNG)
#
# Example:
#   julia ForwardSynthesis.jl \
#     --mode ml \
#     --atom /path/to/atom.h6_tiago2.yaml \
#     --mesh /path/to/mesh.en024048_hion \
#     --atmos /path/to/atm3d.en024048_hion \
#     --pred_h5 sunnynet_output_3D_sim_s5_3x3.hdf5 \
#     --pred_key populations \
#     --out_h5 intensity_ml_3x3.h5 \
#     --out_prefix diag_ml \
#     --line_index 5 \
#     --x 33 --y 21 \
#     --cmass_n 400 --cmass_logmin -6 --cmass_logmax 2
#
#   julia ForwardSynthesis.jl \
#     --mode bifrost \
#     --atom /path/to/atom.h6_tiago2.yaml \
#     --mesh /path/to/mesh.en024048_hion \
#     --atmos /path/to/atm3d.en024048_hion \
#     --pops /path/to/out_pop \
#     --out_h5 intensity_bifrost.h5 \
#     --out_prefix diag_bifrost \
#     --line_index 5
#
# Notes:
#   - The script keeps array permutations consistent with your notebook.
#   - Departure coefficients are assumed to be stored in an HDF5 dataset
#     named by --pred_key (default: "populations").
# -----------------------------------------------------------------------------

using Muspel
using StaticArrays
using AtomicData
using HDF5
using ProgressMeter
using Base.Threads
using Interpolations
using Plots

# -----------------------------
# Small CLI helper
# -----------------------------
function _getarg(flag::String, default::String="")
    for i in 1:length(ARGS)
        if ARGS[i] == flag
            return i < length(ARGS) ? ARGS[i+1] : default
        end
    end
    return default
end

function _hasflag(flag::String)
    any(==(flag), ARGS)
end

function parse_cli()
    mode        = _getarg("--mode", "ml")  # "ml" or "bifrost"

    atom_file   = _getarg("--atom", "")
    mesh_file   = _getarg("--mesh", "")
    atmos_file  = _getarg("--atmos", "")
    pops_file   = _getarg("--pops", "")     # only needed for mode=bifrost

    pred_h5     = _getarg("--pred_h5", "")  # only needed for mode=ml
    pred_key    = _getarg("--pred_key", "populations")

    out_h5      = _getarg("--out_h5", "intensity_out.h5")
    out_prefix  = _getarg("--out_prefix", "diag")

    line_index  = parse(Int, _getarg("--line_index", "5"))  # your notebook used h_atom.lines[5]
    x_pick      = parse(Int, _getarg("--x", "33"))
    y_pick      = parse(Int, _getarg("--y", "21"))

    cmass_n     = parse(Int, _getarg("--cmass_n", "400"))
    cmass_logmin= parse(Float64, _getarg("--cmass_logmin", "-6"))
    cmass_logmax= parse(Float64, _getarg("--cmass_logmax", "2"))

    # Voigt grid (keep notebook defaults)
    voigt_a_n   = parse(Int, _getarg("--voigt_a_n", "20000"))
    voigt_v_n   = parse(Int, _getarg("--voigt_v_n", "2500"))
    voigt_a_min = parse(Float64, _getarg("--voigt_a_min", "1e-4"))
    voigt_a_max = parse(Float64, _getarg("--voigt_a_max", "1e1"))
    voigt_v_min = parse(Float64, _getarg("--voigt_v_min", "0"))
    voigt_v_max = parse(Float64, _getarg("--voigt_v_max", "500"))

    if atom_file == "" || mesh_file == "" || atmos_file == ""
        error("Missing required inputs. Need --atom, --mesh, --atmos.")
    end
    if mode == "ml" && pred_h5 == ""
        error("mode=ml requires --pred_h5 (predicted departure coefficients HDF5).")
    end
    if mode == "bifrost" && pops_file == ""
        error("mode=bifrost requires --pops (Multi3D out_pop file).")
    end

    return (
        mode=mode,
        atom_file=atom_file,
        mesh_file=mesh_file,
        atmos_file=atmos_file,
        pops_file=pops_file,
        pred_h5=pred_h5,
        pred_key=pred_key,
        out_h5=out_h5,
        out_prefix=out_prefix,
        line_index=line_index,
        x_pick=x_pick,
        y_pick=y_pick,
        cmass_n=cmass_n,
        cmass_logmin=cmass_logmin,
        cmass_logmax=cmass_logmax,
        voigt_a_n=voigt_a_n,
        voigt_v_n=voigt_v_n,
        voigt_a_min=voigt_a_min,
        voigt_a_max=voigt_a_max,
        voigt_v_min=voigt_v_min,
        voigt_v_max=voigt_v_max,
    )
end

# -----------------------------
# Column-mass remapping helpers
# (directly from your notebook, with minor cleanup)
# -----------------------------
function cmass_from_rho(ρ::AbstractVector{T}, z::AbstractVector{T}) where {T<:Real}
    n = length(ρ)
    cm = similar(ρ)
    cm[1] = zero(T)
    @inbounds for i in 2:n
        dz = -(z[i] - z[i-1])  # z increases upward
        cm[i] = cm[i-1] + 0.5*(ρ[i] + ρ[i-1]) * dz
    end
    return cm
end

function interpolate_column(ρ, z, Y, cmass_new; logx::Bool=false, logy::Bool=false)
    cm = cmass_from_rho(ρ, z)

    x    = logx ? log10.(cm)        : cm
    xnew = logx ? log10.(cmass_new) : cmass_new

    m     = length(cmass_new)
    nvars = size(Y, 2)
    out   = Matrix{eltype(Y)}(undef, m, nvars)

    @inbounds for v in 1:nvars
        ycol = view(Y, :, v)
        y    = logy ? log10.(ycol) : ycol

        itp = LinearInterpolation(x, y, extrapolation_bc=Line())
        vals = itp.(xnew)

        out[:, v] .= logy ? 10 .^ vals : vals
    end

    return out
end

"""
ρ_super   :: (nx, ny, nz)
pops4d    :: (nx, ny, nz, nvars)
cmass_new :: (m)

returns   :: (nx, ny, m, nvars)
"""
function interpolate_everything(ρ_super, z, pops4d, cmass_new; logx::Bool=false, logy::Bool=false)
    nx, ny, nz = size(ρ_super)
    @assert size(pops4d,1) == nx
    @assert size(pops4d,2) == ny
    @assert size(pops4d,3) == nz

    nvars = size(pops4d, 4)
    m     = length(cmass_new)
    out   = Array{eltype(pops4d)}(undef, nx, ny, m, nvars)

    @threads for ix in 1:nx
        for iy in 1:ny
            ρ = view(ρ_super, ix, iy, :)
            Y = reshape(view(pops4d, ix, iy, :, :), nz, nvars)  # force (nz, nvars)
            col = interpolate_column(ρ, z, Y, cmass_new; logx=logx, logy=logy)  # (m, nvars)
            @inbounds out[ix, iy, :, :] .= col
        end
    end

    return out
end

function compute_ztop(rho3d, z, mtop)
    nx, ny, nz = size(rho3d)
    ztop = Array{eltype(rho3d)}(undef, nx, ny)
    for ix in 1:nx, iy in 1:ny
        ρ  = view(rho3d, ix, iy, :)
        cm = cmass_from_rho(ρ, z)
        itp = LinearInterpolation(cm, z, extrapolation_bc=Line())
        ztop[ix,iy] = itp(mtop)
    end
    return ztop
end

function invert_cmass(cmass, rho, ztop)
    N = length(cmass)
    z = similar(cmass)
    z[1] = ztop
    @inbounds for i = 2:N
        dm = cmass[i] - cmass[i-1]
        z[i] = z[i-1] - dm / rho[i-1]
    end
    return z
end

function compute_z_from_cmass_3d(rho_new, cmass_new, ztop)
    nx, ny, nz = size(rho_new)
    znew = Array{eltype(rho_new)}(undef, nx, ny, nz)
    for ix in 1:nx, iy in 1:ny
        znew[ix,iy,:] = invert_cmass(cmass_new, view(rho_new, ix, iy, :), ztop[ix,iy])
    end
    return znew
end

function remap_atmosphere_cmass(atmos::Atmosphere3D, new_cmass_scale)
    nx, ny, nz = atmos.nx, atmos.ny, atmos.nz

    # (nz,nx,ny) → (nx,ny,nz)
    temp = permutedims(atmos.temperature,       (2,3,1))
    vx   = permutedims(atmos.velocity_x,        (2,3,1))
    vy   = permutedims(atmos.velocity_y,        (2,3,1))
    vz   = permutedims(atmos.velocity_z,        (2,3,1))
    ne   = permutedims(atmos.electron_density,  (2,3,1))
    nh   = permutedims(atmos.hydrogen1_density, (2,3,1))
    np   = permutedims(atmos.proton_density,    (2,3,1))
    rho  = permutedims(atmos.plasma_density,    (2,3,1))

    ztop = compute_ztop(rho, atmos.z, new_cmass_scale[1])

    f4(A) = reshape(A, nx, ny, nz, 1)
    temp4, vx4, vy4, vz4 = f4(temp), f4(vx), f4(vy), f4(vz)
    ne4, nh4, np4, rho4  = f4(ne),   f4(nh), f4(np), f4(rho)

    temp_new = dropdims(interpolate_everything(rho, atmos.z, temp4, new_cmass_scale; logx=true, logy=false), dims=4)
    vx_new   = dropdims(interpolate_everything(rho, atmos.z, vx4,   new_cmass_scale; logx=true, logy=false), dims=4)
    vy_new   = dropdims(interpolate_everything(rho, atmos.z, vy4,   new_cmass_scale; logx=true, logy=false), dims=4)
    vz_new   = dropdims(interpolate_everything(rho, atmos.z, vz4,   new_cmass_scale; logx=true, logy=false), dims=4)
    ne_new   = dropdims(interpolate_everything(rho, atmos.z, ne4,   new_cmass_scale; logx=true, logy=true),  dims=4)
    nh_new   = dropdims(interpolate_everything(rho, atmos.z, nh4,   new_cmass_scale; logx=true, logy=true),  dims=4)
    np_new   = dropdims(interpolate_everything(rho, atmos.z, np4,   new_cmass_scale; logx=true, logy=true),  dims=4)
    rho_new  = dropdims(interpolate_everything(rho, atmos.z, rho4,  new_cmass_scale; logx=true, logy=true),  dims=4)

    z_new = compute_z_from_cmass_3d(rho_new, new_cmass_scale, ztop)

    # back to Atmosphere3D layout (nz,nx,ny)
    g(A) = permutedims(A, (3,1,2))

    return Atmosphere3D(
        nx, ny, length(z_new),
        atmos.x, atmos.y,
        g(z_new),
        g(temp_new),
        g(vx_new), g(vy_new), g(vz_new),
        g(ne_new), g(nh_new), g(np_new),
        g(rho_new)
    )
end

# -----------------------------
# Populations + diagnostics helpers
# -----------------------------
function lte_pops_saha(h_atom, atmos::Atmosphere3D)
    # same as notebook:
    # pops = Muspel.saha_boltzmann.(Ref(h_atom), T, ne, nH = nh1 + np)
    pops = Muspel.saha_boltzmann.(
        Ref(h_atom),
        atmos.temperature,
        atmos.electron_density,
        atmos.hydrogen1_density .+ atmos.proton_density
    )
    # convert Vector-of-SVectors to Float32 array like notebook
    pops_s = SVector{h_atom.nlevels,Float32}.(pops)
    reint  = reshape(reinterpret(Float32, pops_s), h_atom.nlevels, size(pops_s)...)
    # notebook:
    # pops4d = permutedims(reinterpreted_pops_S, (3,4,2,1))  -> (nx, ny, nz, nlevels)
    pops4d = permutedims(reint, (3,4,2,1))
    return pops4d
end

function load_pred_depcoeff(pred_h5::String, pred_key::String)
    f = h5open(pred_h5, "r")
    raw = HDF5.readmmap(f[pred_key])
    close(f)

    # notebook:
    # dep_coeff = PermutedDimsArray(raw, (3, 4, 2, 1))
    # so assume raw is (nlevels, nz, ny, nx) or similar; this permutation yields (ny?, nx?, ?, ?).
    # We keep EXACT permutation from notebook.
    dep_coeff = PermutedDimsArray(raw, (3, 4, 2, 1))
    return dep_coeff
end

function load_multi3d_pops(pops_file::String, atmos::Atmosphere3D, nlevels::Int)
    pops_out_nlte, pops_out_lte = read_pops_multi3d(pops_file, atmos.nx, atmos.ny, atmos.nz, nlevels)
    return pops_out_nlte, pops_out_lte
end

function remap_pops_to_cmass(atmos::Atmosphere3D, pops4d_nxnyznv, new_cmass_scale; logx=true, logy=true)
    # expects pops in (nx,ny,nz,nvars) OR a PermutedDimsArray that behaves like that
    rho_nxnyz = PermutedDimsArray(atmos.plasma_density, (2, 3, 1))  # (nx,ny,nz)
    pops_new  = interpolate_everything(rho_nxnyz, atmos.z, pops4d_nxnyznv, new_cmass_scale; logx=logx, logy=logy)
    return pops_new
end

# Source function at line center (same algebra you used)
function line_source_function(h_atom, λ0_m::Float64, nltepops_nzmxy; l=2, u=3)
    # nltepops_nzmxy expected shape: (nz, nx, ny, nlevels) or something indexable as [:,:,:,level]
    h = 6.62607015e-34
    c = 2.99792458e8
    ν = c / λ0_m

    n_l = nltepops_nzmxy[:, :, :, l]
    n_u = nltepops_nzmxy[:, :, :, u]
    g_l = h_atom.g[l]
    g_u = h_atom.g[u]

    prefactor = 2h * ν^3 / c^2
    Sν = prefactor ./ ((g_u .* n_l) ./ (g_l .* n_u) .- 1)
    return Sν
end

# -----------------------------
# Synthesis
# -----------------------------
function default_background_atom_files()
    bckgr_atoms = [
        "Al.yaml","C.yaml","Ca.yaml","Fe.yaml","H_6.yaml","He.yaml","KI.yaml","Mg.yaml",
        "N.yaml","Na.yaml","NiI.yaml","O.yaml","S.yaml","Si.yaml",
    ]
    return [joinpath(AtomicData.get_atom_dir(), a) for a in bckgr_atoms]
end

function synthesize_intensity_3d(atms::Atmosphere3D, h_atom, line_index::Int, nltepops_nz_nx_ny_nlev;
                                voigt_cfg=(a_min=1f-4,a_max=1f1,a_n=20000,v_min=0f0,v_max=5f2,v_n=2500))
    my_line = h_atom.lines[line_index]

    a = LinRange(Float32(voigt_cfg.a_min), Float32(voigt_cfg.a_max), voigt_cfg.a_n)
    v = LinRange(Float32(voigt_cfg.v_min), Float32(voigt_cfg.v_max), voigt_cfg.v_n)
    voigt_itp = create_voigt_itp(a, v)

    atom_files = default_background_atom_files()
    σ_itp = get_σ_itp(atms, my_line.λ0, atom_files)

    intensity = Array{Float32,3}(undef, my_line.nλ, atms.ny, atms.nx)
    p = Progress(atms.nx; desc="Synthesis columns (x)")

    # like notebook:
    # n_u = nltepops[:, :, :, 3]
    # n_l = nltepops[:, :, :, 2]
    n_u = nltepops_nz_nx_ny_nlev[:, :, :, 3]
    n_l = nltepops_nz_nx_ny_nlev[:, :, :, 2]

    Threads.@threads for i in 1:atms.nx
        buf = RTBuffer(atms.nz, my_line.nλ, Float32)
        for j in 1:atms.ny
            calc_line_prep!(my_line, buf, atms[:, j, i], σ_itp)
            calc_line_1D!(my_line, buf, my_line.λ, atms[:, j, i],
                          n_u[:, i, j], n_l[:, i, j], voigt_itp)
            intensity[:, j, i] = buf.intensity
        end
        next!(p)
    end

    return (intensity=intensity, wave=my_line.λ, line=my_line)
end

# -----------------------------
# Diagnostics + output
# -----------------------------
function save_intensity_h5(out_h5::String, intensity, wave)
    f = h5open(out_h5, "w")
    f["intensity"] = intensity
    f["wave_HA"]   = wave
    close(f)
end

function plot_diag_depcoeff(out_png::String, cmass_new, pred_dep::AbstractVector, orig_dep::Union{Nothing,AbstractVector})
    x = log10.(cmass_new)
    plt = plot(x, log10.(pred_dep), label="pred dep coeff", color=:black)
    if orig_dep !== nothing
        plot!(plt, x, log10.(orig_dep), label="orig dep coeff", color=:red)
    end
    xlabel!(plt, "log10(cmass)")
    ylabel!(plt, "log10(dep coeff)")
    savefig(plt, out_png)
end

function plot_diag_Snu(out_png::String, cmass_new, S_pred::AbstractVector, S_orig::Union{Nothing,AbstractVector})
    x = log10.(cmass_new)
    plt = plot(x, log10.(S_pred), label="pred Sν", color=:black)
    if S_orig !== nothing
        plot!(plt, x, log10.(S_orig), label="orig Sν", color=:red)
    end
    xlabel!(plt, "log10(cmass)")
    ylabel!(plt, "log10(Sν)  [SI units]")
    savefig(plt, out_png)
end

# -----------------------------
# Main pipeline
# -----------------------------
function main()
    cfg = parse_cli()

    println("Mode        : ", cfg.mode)
    println("Threads     : ", Threads.nthreads())

    println("Reading atom: ", cfg.atom_file)
    h_atom = Muspel.read_atom(cfg.atom_file)

    println("Reading atmos: mesh=", cfg.mesh_file, " atmos=", cfg.atmos_file)
    atmos = read_atmos_multi3d(cfg.mesh_file, cfg.atmos_file)

    # new cmass scale (notebook)
    new_cmass_scale = Float32.(10 .^ range(cfg.cmass_logmin, cfg.cmass_logmax, length=cfg.cmass_n))

    # LTE pops on original grid, then remap to cmass
    println("Computing LTE populations (Saha-Boltzmann) on original grid...")
    pops4d_lte = lte_pops_saha(h_atom, atmos)  # (nx,ny,nz,nlev)

    println("Remapping atmosphere to cmass grid...")
    remapped_atmos = remap_atmosphere_cmass(atmos, new_cmass_scale)

    println("Remapping LTE populations to cmass grid...")
    pops_lte_new = remap_pops_to_cmass(atmos, pops4d_lte, new_cmass_scale; logx=true, logy=true) # (nx,ny,m,nlev)

    # We will build NLTE populations on cmass grid as (nz, nx, ny, nlev) later for Muspel usage.
    nlte_pops_nz_nx_ny_nlev = nothing
    dep_coeff_for_diag = nothing
    orig_dep_for_diag = nothing
    Snu_orig_for_diag = nothing

    if cfg.mode == "ml"
        println("Loading predicted departure coefficients from: ", cfg.pred_h5, "  key=", cfg.pred_key)
        dep_coeff = load_pred_depcoeff(cfg.pred_h5, cfg.pred_key)  # PermutedDimsArray(...)

        # notebook:
        # nlte_populations = dep_coeff .* pops_new
        nlte_pop_nxnyznv = dep_coeff .* pops_lte_new  # shapes should broadcast as in your notebook

        # notebook:
        # reshaped_nlte_pops = permutedims(nlte_populations, (3, 4, 1, 2))
        # permutted_reshaped_nlte_pops = PermutedDimsArray(reshaped_nlte_pops, (1, 3, 4, 2))
        # the final is (nz, nx, ny, nlev)
        reshaped = permutedims(nlte_pop_nxnyznv, (3, 4, 1, 2))
        nlte_pops_nz_nx_ny_nlev = PermutedDimsArray(reshaped, (1, 3, 4, 2))

        # For diagnostics: if we also have the original Multi3D pops file (optional),
        # we can compare pred dep coeff vs orig dep coeff.
        if cfg.pops_file != ""
            println("Also reading original Multi3D pops for diagnostics: ", cfg.pops_file)
            pops_out_nlte, pops_out_lte = load_multi3d_pops(cfg.pops_file, atmos, h_atom.nlevels)

            # remap both to cmass (your notebook permutations):
            # pops_out_lte_new_cmass  = PermutedDimsArray(interpolate_everything(... PermutedDimsArray(pops_out_lte,(2,3,1,4)) ...),(3,1,2,4))
            # That final is (nz, nx, ny, nlev)
            pops_out_lte_new = PermutedDimsArray(
                remap_pops_to_cmass(atmos, PermutedDimsArray(pops_out_lte, (2,3,1,4)), new_cmass_scale; logx=true, logy=true),
                (3,1,2,4)
            )
            pops_out_nlte_new = PermutedDimsArray(
                remap_pops_to_cmass(atmos, PermutedDimsArray(pops_out_nlte, (2,3,1,4)), new_cmass_scale; logx=true, logy=true),
                (3,1,2,4)
            )

            # orig departure coefficients on cmass grid:
            # orig_dep = pops_out_nlte_new[:,x,y,l] ./ pops_out_lte_new[:,x,y,l]
            x = cfg.x_pick
            y = cfg.y_pick
            l = 2
            orig_dep_for_diag = pops_out_nlte_new[:, x, y, l] ./ pops_out_lte_new[:, x, y, l]
            # pred dep coefficient (match your notebook permutation used in plot)
            pred_dep_for_diag = PermutedDimsArray(dep_coeff, (3,1,2,4))[:, x, y, l]
            dep_coeff_for_diag = pred_dep_for_diag

            # Source function comparison (line center λ0)
            λ0_m = Float64(h_atom.lines[cfg.line_index].λ0) # Muspel gives meters? if not, you can override with --λ0
            Sν_orig = line_source_function(h_atom, λ0_m, pops_out_nlte_new; l=2, u=3)
            Sν_pred = line_source_function(h_atom, λ0_m, nlte_pops_nz_nx_ny_nlev; l=2, u=3)
            Snu_orig_for_diag = Sν_orig[:, x, y]  # vector over depth

            # Plot dep coeff + Sν (both comparisons)
            plot_diag_depcoeff(cfg.out_prefix * "_depcoeff.png", new_cmass_scale, dep_coeff_for_diag, orig_dep_for_diag)
            plot_diag_Snu(cfg.out_prefix * "_Snu.png", new_cmass_scale, Sν_pred[:, x, y], Snu_orig_for_diag)
        else
            # Minimal diagnostics: just plot predicted dep coeff at (x,y) for level 2
            x = cfg.x_pick
            y = cfg.y_pick
            l = 2
            dep_coeff_for_diag = PermutedDimsArray(dep_coeff, (3,1,2,4))[:, x, y, l]
            plot_diag_depcoeff(cfg.out_prefix * "_depcoeff.png", new_cmass_scale, dep_coeff_for_diag, nothing)

            λ0_m = Float64(h_atom.lines[cfg.line_index].λ0)
            Sν_pred = line_source_function(h_atom, λ0_m, nlte_pops_nz_nx_ny_nlev; l=2, u=3)
            plot_diag_Snu(cfg.out_prefix * "_Snu.png", new_cmass_scale, Sν_pred[:, x, y], nothing)
        end

    elseif cfg.mode == "bifrost"
        println("Reading original Multi3D NLTE+LTE pops: ", cfg.pops_file)
        pops_out_nlte, pops_out_lte = load_multi3d_pops(cfg.pops_file, atmos, h_atom.nlevels)

        pops_out_nlte_new = PermutedDimsArray(
            remap_pops_to_cmass(atmos, PermutedDimsArray(pops_out_nlte, (2,3,1,4)), new_cmass_scale; logx=true, logy=true),
            (3,1,2,4)
        )
        pops_out_lte_new = PermutedDimsArray(
            remap_pops_to_cmass(atmos, PermutedDimsArray(pops_out_lte, (2,3,1,4)), new_cmass_scale; logx=true, logy=true),
            (3,1,2,4)
        )

        nlte_pops_nz_nx_ny_nlev = pops_out_nlte_new

        # Diagnostics: dep coeff (orig only) and Sν (orig only)
        x = cfg.x_pick
        y = cfg.y_pick
        l = 2
        orig_dep = pops_out_nlte_new[:, x, y, l] ./ pops_out_lte_new[:, x, y, l]
        plot_diag_depcoeff(cfg.out_prefix * "_depcoeff.png", new_cmass_scale, orig_dep, nothing)

        λ0_m = Float64(h_atom.lines[cfg.line_index].λ0)
        Sν = line_source_function(h_atom, λ0_m, nlte_pops_nz_nx_ny_nlev; l=2, u=3)
        plot_diag_Snu(cfg.out_prefix * "_Snu.png", new_cmass_scale, Sν[:, x, y], nothing)

    else
        error("Unknown --mode $(cfg.mode). Use 'ml' or 'bifrost'.")
    end

    println("Synthesizing line profiles (Muspel 1D per column)...")
    syn = synthesize_intensity_3d(
        remapped_atmos,
        h_atom,
        cfg.line_index,
        nlte_pops_nz_nx_ny_nlev;
        voigt_cfg=(
            a_min=Float32(cfg.voigt_a_min),
            a_max=Float32(cfg.voigt_a_max),
            a_n=cfg.voigt_a_n,
            v_min=Float32(cfg.voigt_v_min),
            v_max=Float32(cfg.voigt_v_max),
            v_n=cfg.voigt_v_n
        )
    )

    println("Saving intensity to: ", cfg.out_h5)
    save_intensity_h5(cfg.out_h5, syn.intensity, syn.wave)

    println("Done.")
    println("Wrote plots: $(cfg.out_prefix)_depcoeff.png and $(cfg.out_prefix)_Snu.png")
end

# Run
main()