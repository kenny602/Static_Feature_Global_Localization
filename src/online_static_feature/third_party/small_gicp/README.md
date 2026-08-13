# small_gicp (vendored subset)

이 디렉토리는 외부 라이브러리를 워크스페이스에 동봉한 것이다. 직접 수정하지 않는다.
업스트림에 반영할 변경이 생기면 아래 저장소에 PR을 올리고 여기를 갱신한다.

- Upstream: https://github.com/koide3/small_gicp
- Commit:   8e665814a9fcbc3b8a3e5bde135917f586abed6e (2024-09-24)
- License:  MIT (`LICENSE` 참조)

## 왜 동봉했나

`SMALL_GICP_ROOT`가 개발 머신의 절대경로(`/home/rain/small_gicp_ws/small_gicp`)를
가리키고 있어서, 다른 머신에서는 clone만으로 빌드가 되지 않았다. small_gicp는
거의 header-only라 필요한 부분이 336KB에 불과하므로 submodule 대신 동봉을 택했다.
`git clone --recursive`를 빼먹어 빌드가 깨지는 경우를 없애는 것이 목적이다.

## 포함 범위

빌드에 실제로 필요한 것만 가져왔다. upstream의 `data/`, `docs/`, `src/test`,
`src/example`, `src/benchmark`, python 바인딩은 제외했다.

- `include/small_gicp/` — 헤더 전체
- `src/small_gicp/registration/registration.cpp`
- `src/small_gicp/registration/registration_helper.cpp`
  (두 파일은 라이브러리로 링크하지 않고 상위 CMakeLists가 각 노드 타겟에
  직접 컴파일해 넣는다)
- `LICENSE`

## 업스트림 갱신 방법

```bash
git clone https://github.com/koide3/small_gicp /tmp/small_gicp
cd /tmp/small_gicp && git checkout <new-sha>
DST=<ws>/src/online_static_feature/third_party/small_gicp
rm -rf "$DST/include" && cp -r include "$DST/include"
cp src/small_gicp/registration/registration.cpp \
   src/small_gicp/registration/registration_helper.cpp \
   "$DST/src/small_gicp/registration/"
cp LICENSE "$DST/LICENSE"
```

갱신 후 이 파일의 Commit 줄을 바꾼다.
