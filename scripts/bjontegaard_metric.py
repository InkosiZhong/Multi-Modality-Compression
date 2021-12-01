import numpy as np
import scipy.interpolate

def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), PSNR1[np.argsort(lR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), PSNR2[np.argsort(lR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2-int1)/(max_int-min_int)

    return avg_diff


def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2-int1)/(max_int-min_int)
    avg_diff = (np.exp(avg_exp_diff)-1)*100
    return avg_diff


if __name__ == '__main__':
    # rgb
    print('rgb (FLIR)')
    bpp_ours = np.array([0.0547, 0.0826, 0.1238, 0.1792, 0.2513])
    psnr_ours = np.array([35.092, 36.626, 38.103, 39.471, 40.704])

    bpp_bpg = np.array([0.08254, 0.12622, 0.19486, 0.30056])
    psnr_bpg = np.array([35.377, 36.877, 38.412, 40.032])

    bpp_minnen = np.array([0.0576, 0.0869, 0.131, 0.1929, 0.2689])
    psnr_minnen = np.array([34.972, 36.435, 37.919, 39.299, 40.606])

    print('ours vs bpg')
    print('BD-PSNR: ', BD_PSNR(bpp_bpg, psnr_bpg, bpp_ours, psnr_ours))
    print('BD-RATE: ', BD_RATE(bpp_bpg, psnr_bpg, bpp_ours, psnr_ours))

    print('ours vs minnen')
    print('BD-PSNR: ', BD_PSNR(bpp_minnen, psnr_minnen, bpp_ours, psnr_ours))
    print('BD-RATE: ', BD_RATE(bpp_minnen, psnr_minnen, bpp_ours, psnr_ours))

    print('minnen vs bpg')
    print('BD-PSNR: ', BD_PSNR(bpp_bpg, psnr_bpg, bpp_minnen, psnr_minnen))
    print('BD-RATE: ', BD_RATE(bpp_bpg, psnr_bpg, bpp_minnen, psnr_minnen))

    print('rgb (KAIST)')
    bpp_ours = np.array([0.0864, 0.12, 0.1699, 0.2389, 0.336])
    psnr_ours = np.array([33.751, 35.344, 36.939, 38.359, 39.838])

    bpp_bpg = np.array([0.33392, 0.23219, 0.15530, 0.10642])
    psnr_bpg = np.array([39.302, 37.374, 35.452, 33.746])

    bpp_minnen = np.array([0.0811, 0.1162, 0.1694, 0.2469, 0.351])
    psnr_minnen = np.array([33.222, 34.574, 36.07, 37.728, 38.988])

    print('ours vs bpg')
    print('BD-PSNR: ', BD_PSNR(bpp_bpg, psnr_bpg, bpp_ours, psnr_ours))
    print('BD-RATE: ', BD_RATE(bpp_bpg, psnr_bpg, bpp_ours, psnr_ours))

    print('ours vs minnen')
    print('BD-PSNR: ', BD_PSNR(bpp_minnen, psnr_minnen, bpp_ours, psnr_ours))
    print('BD-RATE: ', BD_RATE(bpp_minnen, psnr_minnen, bpp_ours, psnr_ours))

    print('minnen vs bpg')
    print('BD-PSNR: ', BD_PSNR(bpp_bpg, psnr_bpg, bpp_minnen, psnr_minnen))
    print('BD-RATE: ', BD_RATE(bpp_bpg, psnr_bpg, bpp_minnen, psnr_minnen))

    # ir
    print('ir (FLIR)')
    bpp_ours = [0.0923, 0.1384, 0.2013, 0.3097]
    psnr_ours = [32.283, 33.382, 34.378, 35.39]

    bpp_bpg = [0.1188, 0.155, 0.2048, 0.301]
    psnr_bpg = [32.272, 33.034, 33.799, 34.74]

    bpp_minnen = [0.0986, 0.1436, 0.2117, 0.318]
    psnr_minnen = [32.148, 33.262, 34.323, 35.323]

    print('ours vs bpg')
    print('BD-PSNR: ', BD_PSNR(bpp_bpg, psnr_bpg, bpp_ours, psnr_ours))
    print('BD-RATE: ', BD_RATE(bpp_bpg, psnr_bpg, bpp_ours, psnr_ours))

    print('ours vs minnen')
    print('BD-PSNR: ', BD_PSNR(bpp_minnen, psnr_minnen, bpp_ours, psnr_ours))
    print('BD-RATE: ', BD_RATE(bpp_minnen, psnr_minnen, bpp_ours, psnr_ours))

    print('minnen vs bpg')
    print('BD-PSNR: ', BD_PSNR(bpp_bpg, psnr_bpg, bpp_minnen, psnr_minnen))
    print('BD-RATE: ', BD_RATE(bpp_bpg, psnr_bpg, bpp_minnen, psnr_minnen))

    print('ir (KAIST)')
    bpp_ours = [0.0635, 0.0883, 0.1254, 0.1635, 0.2354]
    psnr_ours = [39.813, 41.556, 42.994, 44.42, 45.915]

    bpp_bpg = [0.26553, 0.19053, 0.13779, 0.10004, 0.07084]
    psnr_bpg = [44.907, 43.753, 42.487, 41.107, 39.593]

    bpp_minnen = [0.0657, 0.0927, 0.1339, 0.1891, 0.2551]
    psnr_minnen = [39.256, 41.009, 42.779, 44.304, 45.604]

    print('ours vs bpg')
    print('BD-PSNR: ', BD_PSNR(bpp_bpg, psnr_bpg, bpp_ours, psnr_ours))
    print('BD-RATE: ', BD_RATE(bpp_bpg, psnr_bpg, bpp_ours, psnr_ours))

    print('ours vs minnen')
    print('BD-PSNR: ', BD_PSNR(bpp_minnen, psnr_minnen, bpp_ours, psnr_ours))
    print('BD-RATE: ', BD_RATE(bpp_minnen, psnr_minnen, bpp_ours, psnr_ours))

    print('minnen vs bpg')
    print('BD-PSNR: ', BD_PSNR(bpp_bpg, psnr_bpg, bpp_minnen, psnr_minnen))
    print('BD-RATE: ', BD_RATE(bpp_bpg, psnr_bpg, bpp_minnen, psnr_minnen))

    # video
    print('video')
    ours_bpp = [0.0779, 0.1165, 0.1877, 0.2879]
    ours_psnr = [35.522, 37.115, 39.219, 41.444]
        
    fvc_bpp = [0.065, 0.1083, 0.1774, 0.2974]
    fvc_psnr = [34.791, 36.443, 38.034, 40.594]

    h265_bpp = [0.1111, 0.176, 0.2777]
    h265_psnr = [34.712, 35.989, 37.367]
    print('ours vs fvc')
    print('BD-PSNR: ', BD_PSNR(fvc_bpp, fvc_psnr, ours_bpp, ours_psnr))
    print('BD-RATE: ', BD_RATE(fvc_bpp, fvc_psnr, ours_bpp, ours_psnr))

    print('ours vs h265')
    print('BD-PSNR: ', BD_PSNR(h265_bpp, h265_psnr, ours_bpp, ours_psnr))
    print('BD-RATE: ', BD_RATE(h265_bpp, h265_psnr, ours_bpp, ours_psnr))

    # ablation
    print('ablation')
    ours_bpp = [0.12, 0.1699, 0.2389, 0.336]
    ours_psnr = [35.344, 36.939, 38.359, 39.838]

    ca_bpp = [0.1226, 0.1765, 0.2502, 0.3473]
    ca_psnr = [35.125, 36.615, 38.243, 39.629]

    res_bpp = [0.1252, 0.1781, 0.2513, 0.3539]
    res_psnr = [34.996, 36.48, 38.19, 39.505]

    sa_bpp = [0.1087, 0.1605, 0.2326, 0.3301]
    sa_psnr = [34.741, 36.147, 37.703, 38.901]

    cat_bpp = [0.1149, 0.1674, 0.2454, 0.3485]
    cat_psnr = [34.632, 36.074, 37.649, 38.987]

    minnen_bpp = [0.1162, 0.1694, 0.2469, 0.351]
    minnen_psnr = [34.574, 36.07, 37.728, 38.988]

    elem_bpp = [0.2576, 0.3562,]
    elem_psnr = [29.948, 30.879]

    print('ca vs minnen')
    print('BD-PSNR: ', BD_PSNR(minnen_bpp, minnen_psnr, ca_bpp, ca_psnr))
    print('BD-RATE: ', BD_RATE(minnen_bpp, minnen_psnr, ca_bpp, ca_psnr))

    print('sa vs minnen')
    print('BD-PSNR: ', BD_PSNR(minnen_bpp, minnen_psnr, sa_bpp, sa_psnr))
    print('BD-RATE: ', BD_RATE(minnen_bpp, minnen_psnr, sa_bpp, sa_psnr))

    print('ours vs minnen')
    print('BD-PSNR: ', BD_PSNR(minnen_bpp, minnen_psnr, ours_bpp, ours_psnr))
    print('BD-RATE: ', BD_RATE(minnen_bpp, minnen_psnr, ours_bpp, ours_psnr))

    print('ca vs res')
    print('BD-PSNR: ', BD_PSNR(res_bpp, res_psnr, ca_bpp, ca_psnr))
    print('BD-RATE: ', BD_RATE(res_bpp, res_psnr, ca_bpp, ca_psnr))

    sa6_bpp = [0.1135, 0.1645, 0.2358, 0.3282]
    sa6_psnr = [34.933, 36.485, 37.805, 38.958]

    sa1_bpp = [0.1157, 0.1679, 0.2416, 0.3352]
    sa1_psnr = [34.918, 36.271, 37.674, 38.91]

    fcat_bpp = [0.1187, 0.1711, 0.2585, 0.3496]
    fcat_psnr = [34.742, 36.119, 37.97, 39.034]

    print('sa6 vs minnen')
    print('BD-PSNR: ', BD_PSNR(minnen_bpp, minnen_psnr, sa6_bpp, sa6_psnr))
    print('BD-RATE: ', BD_RATE(minnen_bpp, minnen_psnr, sa6_bpp, sa6_psnr))

    print('sa1 vs minnen')
    print('BD-PSNR: ', BD_PSNR(minnen_bpp, minnen_psnr, sa1_bpp, sa1_psnr))
    print('BD-RATE: ', BD_RATE(minnen_bpp, minnen_psnr, sa1_bpp, sa1_psnr))

    print('featcat vs minnen')
    print('BD-PSNR: ', BD_PSNR(minnen_bpp, minnen_psnr, fcat_bpp, fcat_psnr))
    print('BD-RATE: ', BD_RATE(minnen_bpp, minnen_psnr, fcat_bpp, fcat_psnr))