#include <dlfcn.h>
#include <iostream>

int main() {
  void* lib = dlopen("libsim2sim.so", RTLD_LAZY | RTLD_GLOBAL);
  if (!lib) {
    std::cerr << "dlopen: " << dlerror() << "\n";
    return 1;
  }
  using RunFn = int (*)();
  auto run = reinterpret_cast<RunFn>(dlsym(lib, "run"));
  if (!run) {
    std::cerr << "dlsym: " << dlerror() << "\n";
    dlclose(lib);
    return 1;
  }
  int ret = run();
  dlclose(lib);
  return ret;
}
