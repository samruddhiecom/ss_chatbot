[33mcommit 5998b1a9dd866dcae896425e18bf439a37bd280d[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmain[m[33m, [m[1;31morigin/main[m[33m)[m
Author: samsadar236 <your_github_email@example.com>
Date:   Tue Oct 13 16:53:31 2026 +0530

    fix: force cache bust for advice-seeking deploy

 app/rails/input_rail.py | 3 [32m++[m[31m-[m
 1 file changed, 2 insertions(+), 1 deletion(-)

[33mcommit 4fdfd6680da3acc90196207c991ebf4706ba9491[m
Author: samsadar236 <your_github_email@example.com>
Date:   Tue Oct 13 16:48:52 2026 +0530

    feat: advice-seeking deflection - redirect to discovery instead of scoping

 app/rails/input_rail.py | 30 [32m+++++++++++++++++++++++++++++[m[31m-[m
 app/schemas.py          |  4 [32m+++[m[31m-[m
 2 files changed, 32 insertions(+), 2 deletions(-)

[33mcommit 6f1c631fbb43ac632b649ad1e4bfc5e1ba645301[m
Author: samsadar236 <your_github_email@example.com>
Date:   Tue Oct 13 16:40:00 2026 +0530

    fix: hard floor MIN_DISCOVERY_TURNS before any CTA fires

 app/graph.py | 7 [32m++++[m[31m---[m
 1 file changed, 4 insertions(+), 3 deletions(-)
