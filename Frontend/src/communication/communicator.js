// 
import axios from 'axios'
let server_address = 'http://127.0.0.1:14450'

export function getTabularDataset(tabularDataList, getTabularDataCallback) {
    let formData = {"tabularData": tabularDataList}
    console.log('getTabularDataset', tabularDataList);
    axios({
        methods: 'get',
        url: server_address + '/tabulardata',
        params: formData,
        timeout: 5000
    })
    .then((res) => {
        let tabularDatasetList = res['data']
        // console.log("tabularDatasetList", tabularDatasetList)
        getTabularDataCallback(tabularDatasetList)
    })
}

export function getRLDataset(getRLDataCallback) {
    axios({
        methods: 'get',
        url: server_address + '/rldata',
        timeout: 5000
    })
    .then((res) => {
        let RLDatasetList = res['data']
        // console.log('RLDatasetList', RLDatasetList)
        getRLDataCallback(RLDatasetList)
    })
}

// export function getDFList(getDFListCallback) {
//     axios({
//         methods: 'get',
//         url: server_address + '/dflist',
//         timeout: 5000
//     })
//     .then((res) => {
//         let DFListsetList = res['data']
//         getDFListCallback(DFListsetList)
//     })
// }

export function geAlternativeDataset(name, geAlternativeDatasetCallback) {
    // console.log("geAlternativeDataset", name);
    axios({
        methods: 'get',
        url: server_address + '/alternative_data',
        params: {"name": name},
        timeout: 5000
    })
    .then((res) => {
        let AlternativeDataset = res['data']
        // console.log('tabularDatasetList', tabularDatasetList)
        geAlternativeDatasetCallback(AlternativeDataset)
    })
}

// export function getRefreshedData(name, refreshedDataCallback) {
//     axios({
//         methods: 'get',
//         url: server_address + '/refreshdata',
//         params: {"name": name},
//         timeout: 5000
//     })
//     .then((res) => {
//         let tabularDataList = res['data']
//         // console.log('tabularDatasetList', tabularDataList)
//         // console.log("resssss", res['data']['data'])
//         refreshedDataCallback(tabularDataList)
//     })
// }

export function getUploadData(name, uploadDataCallback) {
    axios({
        methods: 'get',
        url: server_address + '/uploadtabulardata',
        params: {"name": name},
        timeout: 5000
    })
    .then((res) => {
        let tabularDataList = res['data']
        // console.log('tabularDatasetList', tabularDataList)
        // console.log("resssss", res['data']['data'])
        uploadDataCallback(tabularDataList)
    })
}



// export function getRecommendedConfig(name, recommendedConfigCallback) {
//     axios({
//         methods: 'get',
//         url: server_address + '/recommendedConfig',
//         params: {name: name},
//         timeout: 5000
//     })
//     .then((res) => {
//         recommendedConfigCallback(res)
//     })
// }


// export async function getPandasData(getPandasDataCallback) {
//     await axios({
//         methods: 'get',
//         url: server_address + '/pandas',
//     })
//     .then((res) => {
//         let pandasDataList = res['data']['data']
//         console.log('pandasList', pandasDataList)
//         getPandasDataCallback(pandasDataList)
//     })
// }