using Ferrite

# Setting `do_apply_zero_r=false` makes solution not converge, but the result is correct
# Setting `do_apply_zero_r=true` makes solution converge, but to the wrong result
do_apply_zero_r = true
do_apply_zero_da = true # If false, affine constraints are not fulfilled (but still converges)
use_affine = true
TOL = 1.e-10

grid = generate_grid(Line, (2,))
addfaceset!(grid, "center", x->x[1]≈0.0)

dh = DofHandler(grid)
push!(dh, :u, 1)
close!(dh)

eval_norm(r, ch) = sqrt(sum(i->r[i]^2, Ferrite.free_dofs(ch)))

function doassemble!(K, r, dh, a)
    # Spring elements 
    k = 1.0
    Ke = [k -k; -k k]
    # Quick and dirty assem
    assembler = start_assemble(K, r)
    for cell in CellIterator(dh)
        ae = a[celldofs(cell)]
        re = Ke*ae
        assemble!(assembler, celldofs(cell), Ke, re)
    end
    r[3] -= 1 # Force on the main dof (right side)
end

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfaceset(grid, "center"), (x,t)->Vec{1}((0.0,))))

if use_affine
    ac = AffineConstraint(1, [3=>-1.0], 0.1) # Symmetric stretching 
    add!(ch, ac)
end

close!(ch)

K = create_sparsity_pattern(dh, ch)
r = zeros(ndofs(dh))

a = zeros(ndofs(dh))
update!(ch, 0.0)

apply!(a, ch) 
@show a
@info "Nonlinear solution"
for niter = 0:2
    doassemble!(K, r, dh, a)
    apply_zero!(K, r, ch)
    Δa = -K\r 
    #apply_zero!(r, ch) #Not needed in this case, but maybe sometimes
    apply_zero!(Δa, ch)
    a .+= Δa
    @show norm(r) #No need to use free_dofs(ch) since constrained dofs have been zerod out
end
@show a

@info "Linear solution"
a_ = zero(a)
doassemble!(K, r, dh, a_)
apply!(K, r, ch)
a_ = -K\r 
apply!(a_, ch)
@show a_
nothing