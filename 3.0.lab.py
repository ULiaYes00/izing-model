import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from numba import njit
import warnings
warnings.filterwarnings('ignore')

N = 60
J = 1.0
h = 0.0
kB = 1.0

N_EQUIL  = int(1000 * N*N * max(1, N / 20))
N_MEAS   = int(800  * N*N * max(1, N / 20))
N_RUNS   = 3  
TEMPS    = np.linspace(1.0, 4.0, 30)

T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))


@njit(cache=True)
def _metropolis_sweep(S, T, J, h, kB, n_steps):
    N = S.shape[0]
    inv_kBT = 1.0 / (kB * T)

    for _ in range(n_steps):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

        neighbors_sum = (
            S[(i + 1) % N, j] +
            S[(i - 1) % N, j] +
            S[i, (j + 1) % N] +
            S[i, (j - 1) % N]
        )

        dE = 2.0 * S[i, j] * (J * neighbors_sum + h)

        if dE <= 0.0 or np.random.random() < np.exp(-dE * inv_kBT):
            S[i, j] *= -1

@njit(cache=True)
def _gibbs_sweep(S, T, J, h, kB, n_steps):
    N = S.shape[0]
    inv_kBT = 1.0 / (kB * T)

    for _ in range(n_steps):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

        H_loc = J * (
            S[(i + 1) % N, j] +
            S[(i - 1) % N, j] +
            S[i, (j + 1) % N] +
            S[i, (j - 1) % N]
        ) + h

        p_plus = 1.0 / (1.0 + np.exp(-2.0 * H_loc * inv_kBT))
        S[i, j] = 1 if np.random.random() < p_plus else -1


@njit(cache=True)
def _total_energy(S, J, h):
    N = S.shape[0]
    E = 0.0
    for i in range(N):
        for j in range(N):
            right = S[i, (j + 1) % N]
            down  = S[(i + 1) % N, j]
            E -= J * S[i, j] * (right + down)
            E -= h * S[i, j]
    return E

@njit(cache=True)
def _measure_loop(S, T, J, h, kB, n_sweeps, algorithm):
    N2 = S.shape[0] ** 2

    energies = np.empty(n_sweeps)
    mags     = np.empty(n_sweeps)

    for sweep in range(n_sweeps):
        if algorithm == 0:
            _metropolis_sweep(S, T, J, h, kB, N2)
        else:
            _gibbs_sweep(S, T, J, h, kB, N2)

        energies[sweep] = _total_energy(S, J, h) / N2
        mags[sweep]     = abs(np.sum(S) / N2)

    return energies, mags


def warmup_numba():
    print("Компіляція numba...", end=' ', flush=True)
    S_tmp = np.random.choice(np.array([-1, 1]), size=(4, 4)).astype(np.int32)
    _metropolis_sweep(S_tmp.copy(), 2.0, 1.0, 0.0, 1.0, 10)
    _gibbs_sweep(S_tmp.copy(), 2.0, 1.0, 0.0, 1.0, 10)
    _measure_loop(S_tmp.copy(), 2.0, 1.0, 0.0, 1.0, 3, 0)
    print("готово")


def init_spins(N):
    return np.random.choice(np.array([-1, 1]), size=(N, N)).astype(np.int32)


def measure(S, T, n_meas, algorithm_id):
    n_sweeps = max(1, n_meas // (N * N))
    energies, mags = _measure_loop(S, T, J, h, kB, n_sweeps, algorithm_id)

    E  = np.mean(energies)
    M  = np.mean(mags)
    E2 = np.mean(energies ** 2)
    N2 = N * N

    C = (E2 - E**2) * N2 / (kB * T**2)
    return E, M, C


def compute_observables(algorithm_name, temps=TEMPS):
    alg_id = 0 if algorithm_name == 'metropolis' else 1
    E_arr, M_arr, C_arr = [], [], []

    print(f"\n{algorithm_name}, N={N}x{N}, runs={N_RUNS}")

    for T in temps:
        E_runs, M_runs, C_runs = [], [], []

        for _ in range(N_RUNS):
            S = init_spins(N)
            if alg_id == 0:
                _metropolis_sweep(S, T, J, h, kB, N_EQUIL)
            else:
                _gibbs_sweep(S, T, J, h, kB, N_EQUIL)
            E, M, C = measure(S, T, N_MEAS, alg_id)
            E_runs.append(E); M_runs.append(M); C_runs.append(C)

        E_arr.append(np.mean(E_runs))
        M_arr.append(np.mean(M_runs))
        C_arr.append(np.mean(C_runs))
        print(f"  T={T:.2f}  E={E_arr[-1]:+.4f}  |M|={M_arr[-1]:.4f}  C={C_arr[-1]:.4f}", flush=True)

    return np.array(E_arr), np.array(M_arr), np.array(C_arr)


def plot_configurations():
    T_list = [1.0, 2.0, T_C, 3.0, 4.0]
    labels = ['T=1.0\n(впоряд.)', 'T=2.0', f'T≈Tc\n({T_C:.2f})',
              'T=3.0', 'T=4.0\n(безлад.)']

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5), facecolor='#0d0d0d')
    fig.suptitle('Конфігурації решітки при різних температурах (Метрополіс)',
                 color='white', fontsize=12)

    for ax, T, lbl in zip(axes, T_list, labels):
        S = init_spins(N)
        _metropolis_sweep(S, T, J, h, kB, N_EQUIL)
        ax.imshow(S, cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title(lbl, color='#cccccc', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('ising_configurations.png', dpi=130,
                facecolor='#0d0d0d', bbox_inches='tight')
    plt.show()


def animate_evolution(T_vis=None, algorithm='metropolis', n_frames=80):
    if T_vis is None:
        T_vis = T_C
    alg_id = 0 if algorithm == 'metropolis' else 1
    S  = init_spins(N)
    N2 = N * N
    sweeps_per_frame = 5

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor='#0d0d0d')

    alg_label = 'Metropolis' if alg_id == 0 else 'Gibbs'
    fig.suptitle(
        f'Еволюція моделі Ізінга | {alg_label} | T={T_vis:.3f} (≈Tc)',
        color='white', fontsize=13)

    ax_spin = axes[0]
    ax_spin.set_title('Спінова конфігурація', color='#aaaaaa', pad=6)
    ax_spin.axis('off')
    im = ax_spin.imshow(S, cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')

    ax_ts = axes[1]
    ax_ts.set_facecolor('#1a1a1a')
    ax_ts.set_title('|M| у часі', color='#aaaaaa', pad=6)
    ax_ts.set_xlabel('Sweeps', color='#888888')
    ax_ts.set_ylabel('|Намагніченість|', color='#888888')
    ax_ts.tick_params(colors='#666666')
    for sp in ax_ts.spines.values():
        sp.set_edgecolor('#333333')

    mag_history = []

    def update(frame):
        if alg_id == 0:
            _metropolis_sweep(S, T_vis, J, h, kB, sweeps_per_frame * N2)
        else:
            _gibbs_sweep(S, T_vis, J, h, kB, sweeps_per_frame * N2)

        im.set_data(S.copy())
        mag_history.append(abs(float(np.mean(S))))

        ax_ts.clear()
        ax_ts.set_facecolor('#1a1a1a')
        ax_ts.set_title('|M| у часі', color='#aaaaaa', pad=6)
        ax_ts.set_xlabel('Sweeps', color='#888888')
        ax_ts.set_ylabel('|Намагніченість|', color='#888888')
        ax_ts.tick_params(colors='#666666')
        for sp in ax_ts.spines.values():
            sp.set_edgecolor('#333333')
        ax_ts.plot(mag_history, color='#e06c75', lw=1.5)
        ax_ts.set_xlim(0, max(10, len(mag_history)))
        ax_ts.set_ylim(0, 1.05)

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=80, blit=False)
    plt.tight_layout()
    ani.save('ising_evolution.gif', writer='pillow', fps=12,
             savefig_kwargs={'facecolor': '#0d0d0d'})
    plt.show()


COLORS = {'metropolis': '#e06c75', 'gibbs': '#61afef', 'tc': '#e5c07b'}


def plot_thermodynamics(res_m, res_g, temps):
    E_m, M_m, C_m = res_m
    E_g, M_g, C_g = res_g

    fig = plt.figure(figsize=(14, 5), facecolor='#111827')
    gs  = GridSpec(1, 3, figure=fig, hspace=0.42, wspace=0.32)

    def style_ax(ax):
        ax.set_facecolor('#1f2937')
        ax.tick_params(colors='#6b7280', labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor('#374151')
        ax.axvline(T_C, color=COLORS['tc'], lw=1.2, ls='--', alpha=0.7,
                   label=f'$T_c$ ≈ {T_C:.3f}')

    datasets = [
        (E_m, E_g, '⟨E⟩ / спін', 'Середня енергія',  gs[0, 0]),
        (M_m, M_g, '⟨|M|⟩',      'Намагніченість',    gs[0, 1]),
        (C_m, C_g, 'C / спін',    'Теплоємність',      gs[0, 2]),
    ]

    for ym, yg, ylabel, title, pos in datasets:
        ax = fig.add_subplot(pos)
        style_ax(ax)
        ax.plot(temps, ym, 'o-', color=COLORS['metropolis'], lw=1.8,
                ms=4, label='Метрополіс')
        ax.plot(temps, yg, 's--', color=COLORS['gibbs'], lw=1.8,
                ms=4, label='Гіббс')
        ax.set_xlabel('Температура T', color='#9ca3af', fontsize=11)
        ax.set_ylabel(ylabel, color='#9ca3af', fontsize=11)
        ax.set_title(title, color='#f3f4f6', fontsize=12,
                     pad=8, fontweight='bold')
        ax.legend(fontsize=9, framealpha=0.3, labelcolor='#d1d5db',
                  facecolor='#111827', edgecolor='#374151')

    fig.suptitle(
        f'Термодинамічні властивості моделі Ізінга (N={N}×{N}, J={J}, h={h})',
        color='#f9fafb', fontsize=14, fontweight='bold', y=1.01)

    plt.savefig('ising_thermodynamics.png', dpi=150,
                facecolor='#111827', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    warmup_numba()

    plot_configurations()
    animate_evolution(T_vis=T_C, algorithm='metropolis', n_frames=60)

    res_metropolis = compute_observables('metropolis', TEMPS)
    res_gibbs      = compute_observables('gibbs', TEMPS)

    plot_thermodynamics(res_metropolis, res_gibbs, TEMPS)